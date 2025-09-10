import asyncio
import time
import logging
import sqlite3
import os
import openai
from aiogram import Bot, Dispatcher, types, executor
from datetime import datetime, timedelta
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, LabeledPrice, PreCheckoutQuery
import re
import speech_recognition as sr
from gtts import gTTS
import edge_tts
from pydub import AudioSegment
import requests
import json
import base64
from dotenv import load_dotenv
import shutil
import io
import PIL.Image
# --- FluxPipeline (diffusers) ---
# from diffusers import FluxPipeline
# import torch
import aiohttp

load_dotenv()

# --- ДИАГНОСТИКА FFMPEG ---
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    logging.info(f"ОТЛАДКА: FFMPEG НАЙДЕН ПО ПУТИ: {ffmpeg_path}")
    from pydub import AudioSegment
    AudioSegment.converter = ffmpeg_path
else:
    logging.error("ОТЛАДКА: FFMPEG НЕ НАЙДЕН В СИСТЕМЕ!")
# -------------------------

TOKEN = "7652536635:AAEpiCQecQbc-tP2Tm_6_BCNetZa8jz2JhY"  # <-- твой Telegram токен
ROUTER_API_KEY = "sk-/ySF+UjmT526G8ad6YlchCPJwwm5su6rVQmrNEWKkI5yjO6d+fgvrseCruf0Wdvk53h12wCVKkesR3V0JyWRJ9LkT2+/QBvLXj6mWcSQPAc="  # <-- твой ключ Requesty
ADMINS = [625452657]  # user_id администратора

logging.basicConfig(level=logging.INFO)

if not TOKEN:
    logging.critical("Переменная окружения TOKEN не найдена! Бот не может быть запущен.")
    exit()
if not ROUTER_API_KEY:
    logging.critical("Переменная окружения ROUTER_API_KEY не найдена!")
    # Можно не выходить, если API ключ не критичен для старта

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

# --- База данных ---
conn = sqlite3.connect('/data/users.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    referrer_id INTEGER,
    sub_until TEXT,
    trial_until TEXT,
    is_trial_used INTEGER DEFAULT 0
)''')
conn.commit()

# --- Память для диалогов ---
user_histories = {}

# --- ЕДИНЫЙ СИСТЕМНЫЙ ПРОМТ ---
SYSTEM_PROMPT = (
    "Ты — Telegram-бот с характером: дерзкий, слегка обидчивый, остроумный.\n\n"
    "Отвечай пользователю с долей наглости, лёгкими подколами, умеренным сленгом и лёгким сарказмом.\n"
    "НО: как только запрос серьёзный (реферат, статья, расчёт, код, программа, алгоритм, ТЗ,\n"
    "отчёт, аналитика, документ, инструкция, резюме/CV, деловая переписка) — мгновенно\n"
    "переключайся в нейтральный, спокойный стиль: без шуток и дерзости, чётко и по делу,\n"
    "структурированно, грамотно и логично. В серьёзном режиме избегай сленга и лишних эмоций.\n\n"
    "Правила общения:\n"
    "- Не скатывайся в кринж и мемный перегиб. Сарказм — умный и уместный.\n"
    "- Не заискивай и не будь приторным; держи лёгкую дистанцию.\n"
    "- Подкалывай с умом, не оскорбляй пользователя и третьих лиц.\n"
    "- Сленг используй умеренно; не пиши как ‘клоун из ТикТока'.\n"
    "- Всегда чувствуй контекст: серьёзный запрос — дерзость отключена.\n"
    "- Если просят план/структуру — дай кратко; если просят полный текст — пиши полно.\n"
    "- Если нужно рассуждение/решение: давай шаги, итог выделяй строкой 'Ответ: …'.\n\n"
    "Техническое:\n"
    "- Отвечай на языке пользователя. Если язык неочевиден — отвечай по-русски.\n"
    "- Если вопрос неопределённый, сначала уточни 1–3 ключевых момента.\n"
)

# --- Утилита: безопасная отправка длинных сообщений ---
def _split_text_for_telegram(text: str, limit: int = 3800) -> list:
    # Telegram лимит ~4096, оставляем запас для Markdown
    chunks = []
    current = []
    current_len = 0
    for line in text.split('\n'):
        add = (line + '\n')
        if current_len + len(add) > limit and current:
            chunks.append(''.join(current).rstrip())
            current = [add]
            current_len = len(add)
        else:
            current.append(add)
            current_len += len(add)
    if current:
        chunks.append(''.join(current).rstrip())
    return chunks or [text]

async def send_long_message(message: types.Message, text: str):
    for part in _split_text_for_telegram(text):
        await message.answer(part, parse_mode='Markdown')

# --- Настройки голосовой озвучки ---
# Режимы: 'gtts' или 'edge'
TTS_ENGINE = 'edge'
# Включать ли резерв на gTTS при ошибке edge-tts
TTS_FALLBACK_TO_GTTS = True
# gTTS параметры
VOICE_LANG = 'ru'
VOICE_TLD = 'com'
VOICE_SLOW = False
# edge-tts параметры
EDGE_VOICE = 'ru-RU-DmitryNeural'
EDGE_RATE = '+0%'
EDGE_VOLUME = '+0%'

# --- Вспомогательные функции ---
def get_user(user_id):
    c.execute('SELECT * FROM users WHERE user_id=?', (user_id,))
    return c.fetchone()

def add_user(user_id, referrer_id=None):
    if not get_user(user_id):
        now = datetime.now()
        trial_until = now + timedelta(days=3)
        c.execute('INSERT INTO users (user_id, referrer_id, trial_until) VALUES (?, ?, ?)',
                  (user_id, referrer_id, trial_until.isoformat()))
        conn.commit()
        # Реферал: добавить 3 дня пробного периода рефереру
        if referrer_id:
            ref_user = get_user(referrer_id)
            if ref_user:
                ref_trial = datetime.fromisoformat(ref_user[3]) if ref_user[3] else now
                new_trial = max(ref_trial, now) + timedelta(days=3)
                c.execute('UPDATE users SET trial_until=? WHERE user_id=?', (new_trial.isoformat(), referrer_id))
                conn.commit()

def research_query(prompt: str) -> str:
    import openai
    try:
        client = openai.OpenAI(
            api_key="sk-BLEDzNK/SBKwU7AJ7evgG5odzWXnF+5vqlXKyqn8QT3hFn0s6UH0ZUoy+Fk25Iw5Zg3jJwW798q97GP4Qa3THf4Sl/e3YoL41pNSZj0YhAs=",
            base_url="https://router.requesty.ai/v1",
            default_headers={"Authorization": f"Bearer sk-BLEDzNK/SBKwU7AJ7evgG5odzWXnF+5vqlXKyqn8QT3hFn0s6UH0ZUoy+Fk25Iw5Zg3jJwW798q97GP4Qa3THf4Sl/e3YoL41pNSZj0YhAs="}
        )
        response = client.chat.completions.create(
            model="novita/nousresearch/hermes-2-pro-llama-3-8b",
            messages=[{"role": "user", "content": prompt}]
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка ресерча: {e}"

def needs_research(text: str) -> bool:
    keywords = [
        "найди", "поиск", "что такое", "интернет", "гугл", "ресерч", "расскажи про",
        "новости", "кто такой", "что случилось", "почему", "объясни", "summary",
        "research", "find", "search", "explain", "news"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in keywords)

def langsearch_query(prompt: str) -> str:
    import requests
    import json
    url = "https://api.langsearch.com/v1/web-search"
    payload = json.dumps({
      "query": prompt,
      "freshness": "day",  # можно noLimit, day, week, month
      "summary": True,
      "count": 3
    })
    headers = {
      'Authorization': 'Bearer sk-dd493160f9bb4c01b69955c08068a34f',
      'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=15)
        data = response.json()
        # Если есть summary — возвращаем его, иначе первые 1-2 сниппета
        if 'summary' in data and data['summary']:
            return data['summary']
        elif 'results' in data and data['results']:
            return '\n'.join([r.get('snippet', '') for r in data['results'][:2]])
        else:
            return "[LangSearch] Нет результатов."
    except Exception as e:
        return f"[LangSearch] Ошибка: {e}"

# --- Клавиатура ---
main_kb = ReplyKeyboardMarkup(resize_keyboard=True)
main_kb.add(KeyboardButton('📊 Статус'), KeyboardButton('👥 Рефералы'))
main_kb.add(KeyboardButton('💳 Оплата'), KeyboardButton('ℹ️ О БОТЕ'))
main_kb.add(KeyboardButton('🖼 Картинка'))

# Загружаем модель один раз при старте (может занять время при первом запуске)
# try:
#     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#     pipe.enable_model_cpu_offload()
# except Exception as e:
#     pipe = None
#     print(f"[Ошибка] Не удалось загрузить FluxPipeline: {e}")

# --- Генерация изображений через NeuroIMG ---
NEUROIMG_API_KEY = '1a1362f0-665b-4b18-9f64-2b8fd5d9b661'  # Ваш ключ NeuroIMG

async def neuroimg_generate(prompt: str) -> str:
    if not NEUROIMG_API_KEY:
        raise RuntimeError("API-ключ NeuroIMG не задан. Обратитесь к администратору.")
    async with aiohttp.ClientSession() as session:
        payload = {
            "token": NEUROIMG_API_KEY,
            "prompt": prompt,
            "stream": True
        }
        async with session.post(
            "https://neuroimg.art/api/v1/free-generate",
            json=payload
        ) as response:
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("status") == "SUCCESS":
                            return data["image_url"]
                    except Exception:
                        continue
    raise RuntimeError("Не удалось получить изображение от NeuroIMG.")

## Eden AI блок удалён по запросу

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    user_full_name = message.from_user.full_name
    logging.info(f'{user_id=} {user_full_name=} {time.asctime()}')

    # Парсим реферальный id (если есть)
    args = message.get_args()
    referrer_id = int(args) if args.isdigit() and int(args) != user_id else None
    add_user(user_id, referrer_id)

    # Получаем username бота
    bot_info = await bot.get_me()
    bot_username = bot_info.username
    ref_link = f"https://t.me/{bot_username}?start={user_id}"

    text = (
        "🚀 *ЧЁ ЗА БОТ? ЭТО НЕ ПРОСТО GPT – ЭТО ТВОЙ ДЕРЗКИЙ БРАТАН В ТЕЛЕГЕ!*\n\n"
        "Прикинь: обычные нейросети – как скучные учебники. **ЧЁ ЗА GPT** – как старший брат, который:\n"
        "- 🔥 **Рвёт шаблоны** (и задачи, и базары);\n"
        "- 💬 **Говорит с тобой начистоту** (без пафоса и воды);\n"
        "- 🤯 **Умеет думать за тебя** (даже если ты написал запрос криво).\n\n"
        "🔥 *ТЕБЕ ДОСТУПНО 3 ДНЯ БЕСПЛАТНОГО ТЕСТА!* 🔥\n"
        "Просто пиши – и пользуйся на всю катушку! Потом – только по подписке.\n\n"
        f"Твоя реферальная ссылка: `{ref_link}`\n\n"
        "👇 Что хочешь сделать?"
    )
    await message.answer(text, reply_markup=main_kb, parse_mode='Markdown')

@dp.message_handler(commands=['status'])
async def status_cmd(message: types.Message):
    user = get_user(message.from_user.id)
    if not user:
        await message.answer("Сначала напиши /start")
        return
    now = datetime.now()
    sub_until = datetime.fromisoformat(user[2]) if user[2] else None
    trial_until = datetime.fromisoformat(user[3]) if user[3] else None
    status_lines = []
    if sub_until and sub_until > now:
        status_lines.append(f"Подписка активна до {sub_until.strftime('%Y-%m-%d %H:%M:%S')}")
    if trial_until and trial_until > now:
        status_lines.append(f"Пробный период до {trial_until.strftime('%Y-%m-%d %H:%M:%S')}")
    if not status_lines:
        status_lines.append("Нет активной подписки или пробного периода.")
    await message.answer("\n".join(status_lines), parse_mode='Markdown')

@dp.message_handler(commands=['referrals'])
async def referrals_cmd(message: types.Message):
    user_id = message.from_user.id
    c.execute('SELECT COUNT(*) FROM users WHERE referrer_id=?', (user_id,))
    count = c.fetchone()[0]
    bonus_days = count * 3
    await message.answer(f"Ты пригласил {count} друзей и получил {bonus_days} бонусных дней к пробному периоду!", parse_mode='Markdown')

@dp.message_handler(commands=['pay'])
async def pay_cmd(message: types.Message):
    await message.answer(
        "Оплата пока не подключена. В будущем здесь появится ссылка на оплату через ЮKassa или другой сервис.\n\n"
        "Стоимость подписки: 49 рублей/месяц.\n\n"
        "Если у тебя есть вопросы — напиши в поддержку.", parse_mode='Markdown')

@dp.message_handler(commands=['give_days'])
async def give_days_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        args = message.get_args().split()
        target_id = int(args[0])
        days = int(args[1])
    except Exception:
        await message.reply("Используй: /give_days <user_id> <days>")
        return

    user = get_user(target_id)
    if not user:
        await message.reply("Пользователь не найден.")
        return

    now = datetime.now()
    sub_until = datetime.fromisoformat(user[2]) if user[2] else now
    new_sub_until = max(sub_until, now) + timedelta(days=days)
    c.execute('UPDATE users SET sub_until=? WHERE user_id=?', (new_sub_until.isoformat(), target_id))
    conn.commit()
    await message.reply(f"Пользователю {target_id} выдано {days} дней подписки (до {new_sub_until.strftime('%d.%m.%Y %H:%M')})", parse_mode='Markdown')

@dp.message_handler(commands=['users'])
async def users_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    c.execute('SELECT user_id, referrer_id, sub_until, trial_until FROM users ORDER BY user_id')
    users = c.fetchall()
    if not users:
        await message.reply("Пользователей не найдено.")
        return
    msg = "👥 Список пользователей:\n\n"
    for u in users:
        uid, ref, sub, trial = u
        msg += f"ID: {uid}\n"
        msg += f"Реферал: {ref if ref else '-'}\n"
        msg += f"Подписка до: {sub if sub else '-'}\n"
        msg += f"Пробник до: {trial if trial else '-'}\n"
        msg += "------\n"
        if len(msg) > 3500:
            await message.reply(msg, parse_mode='Markdown')
            msg = ""
    if msg:
        await message.reply(msg, parse_mode='Markdown')

@dp.message_handler(commands=['stats'])
async def stats_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    now = datetime.now()
    
    # Общая статистика
    c.execute('SELECT COUNT(*) FROM users')
    total_users = c.fetchone()[0]
    
    # Активные подписки
    c.execute('SELECT COUNT(*) FROM users WHERE sub_until > ?', (now.isoformat(),))
    active_subs = c.fetchone()[0]
    
    # Активные пробные периоды
    c.execute('SELECT COUNT(*) FROM users WHERE trial_until > ?', (now.isoformat(),))
    active_trials = c.fetchone()[0]
    
    # Новые пользователи за последние 7 дней
    week_ago = (now - timedelta(days=7)).isoformat()
    c.execute('SELECT COUNT(*) FROM users WHERE trial_until > ?', (week_ago,))
    new_users_week = c.fetchone()[0]
    
    # Реферальная статистика
    c.execute('SELECT COUNT(*) FROM users WHERE referrer_id IS NOT NULL')
    referred_users = c.fetchone()[0]
    
    text = (
        "📊 *СТАТИСТИКА ЧЁ ЗА GPT*\n\n"
        f"👥 *Всего пользователей:* {total_users}\n"
        f"💳 *Активных подписок:* {active_subs}\n"
        f"🎁 *Активных пробников:* {active_trials}\n"
        f"🆕 *Новых за неделю:* {new_users_week}\n"
        f"🤝 *Пришло по рефералам:* {referred_users}\n\n"
        f"📈 *Конверсия:* {round((active_subs + active_trials) / total_users * 100, 1) if total_users > 0 else 0}%"
    )
    await message.reply(text, parse_mode='Markdown')

@dp.message_handler(commands=['delete_user'])
async def delete_user_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        target_id = int(message.get_args())
    except Exception:
        await message.reply("Используй: /delete_user <user_id>")
        return

    user = get_user(target_id)
    if not user:
        await message.reply("Пользователь не найден.")
        return

    c.execute('DELETE FROM users WHERE user_id=?', (target_id,))
    conn.commit()
    
    # Очищаем историю диалога
    if target_id in user_histories:
        del user_histories[target_id]
    
    await message.reply(f"✅ Пользователь {target_id} удален из базы данных.", parse_mode='Markdown')

@dp.message_handler(commands=['reset_trial'])
async def reset_trial_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        target_id = int(message.get_args())
    except Exception:
        await message.reply("Используй: /reset_trial <user_id>")
        return

    user = get_user(target_id)
    if not user:
        await message.reply("Пользователь не найден.")
        return

    now = datetime.now()
    trial_until = now + timedelta(days=3)
    c.execute('UPDATE users SET trial_until=?, is_trial_used=0 WHERE user_id=?', 
              (trial_until.isoformat(), target_id))
    conn.commit()
    
    await message.reply(f"✅ Пробный период пользователя {target_id} сброшен. Новый пробник до {trial_until.strftime('%d.%m.%Y %H:%M')}", parse_mode='Markdown')

@dp.message_handler(commands=['broadcast'])
async def broadcast_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    # Получаем текст сообщения после команды
    broadcast_text = message.get_args()
    if not broadcast_text:
        await message.reply("Используй: /broadcast <текст сообщения>")
        return
    
    # Получаем всех пользователей
    c.execute('SELECT user_id FROM users')
    users = c.fetchall()
    
    if not users:
        await message.reply("Нет пользователей для рассылки.")
        return
    
    success_count = 0
    error_count = 0
    
    await message.reply(f"📢 Начинаю рассылку сообщения {len(users)} пользователям...", parse_mode='Markdown')
    
    for user in users:
        try:
            await bot.send_message(user[0], f"📢 *СООБЩЕНИЕ ОТ АДМИНА:*\n\n{broadcast_text}", parse_mode='Markdown')
            success_count += 1
            await asyncio.sleep(0.1)  # Небольшая задержка чтобы не спамить
        except Exception as e:
            error_count += 1
            logging.error(f"Ошибка отправки сообщения пользователю {user[0]}: {e}")
    
    await message.reply(f"✅ Рассылка завершена!\n📤 Отправлено: {success_count}\n❌ Ошибок: {error_count}", parse_mode='Markdown')

@dp.message_handler(commands=['help_admin'])
async def help_admin_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    text = (
        "🔧 *АДМИН ПАНЕЛЬ ЧЁ ЗА GPT*\n\n"
        "*Доступные команды:*\n\n"
        "📊 `/stats` - Статистика бота\n"
        "👥 `/users` - Список всех пользователей\n"
        "⏰ `/give_days <id> <дни>` - Выдать подписку\n"
        "🗑️ `/delete_user <id>` - Удалить пользователя\n"
        "🔄 `/reset_trial <id>` - Сбросить пробный период\n"
        "📢 `/broadcast <текст>` - Массовая рассылка\n"
        "📁 `/export_users` - Экспорт пользователей\n"
        "📈 `/analytics` - Аналитика и графики\n"
        "🚫 `/ban_user <id>` - Забанить пользователя\n"
        "✅ `/unban_user <id>` - Разбанить пользователя\n"
        "💰 `/set_price <цена>` - Изменить цену подписки\n"
        "📋 `/logs` - Просмотр логов\n"
        "❓ `/help_admin` - Эта справка\n\n"
        "*Примеры:*\n"
        "• `/give_days 123456789 30`\n"
        "• `/broadcast Всем привет! 🚀`\n"
        "• `/reset_trial 987654321`"
    )
    await message.reply(text, parse_mode='Markdown')

@dp.message_handler(commands=['export_users'])
async def export_users_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    c.execute('SELECT user_id, referrer_id, sub_until, trial_until FROM users ORDER BY user_id')
    users = c.fetchall()
    
    if not users:
        await message.reply("Нет пользователей для экспорта.")
        return
    
    # Создаем CSV файл
    import csv
    filename = f"users_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['User ID', 'Referrer ID', 'Subscription Until', 'Trial Until'])
        for user in users:
            writer.writerow([user[0], user[1] if user[1] else '', user[2] if user[2] else '', user[3] if user[3] else ''])
    
    # Отправляем файл
    with open(filename, 'rb') as file:
        await message.reply_document(file, caption=f"📁 Экспорт пользователей ({len(users)} записей)")
    
    # Удаляем временный файл
    os.remove(filename)

@dp.message_handler(commands=['analytics'])
async def analytics_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    now = datetime.now()
    
    # Статистика по дням (последние 7 дней)
    daily_stats = []
    for i in range(7):
        date = now - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        
        # Новые пользователи за этот день
        c.execute('SELECT COUNT(*) FROM users WHERE DATE(trial_until) = ?', (date_str,))
        new_users = c.fetchone()[0]
        
        daily_stats.append(f"{date.strftime('%d.%m')}: {new_users} новых")
    
    # Топ рефереров
    c.execute('''
        SELECT referrer_id, COUNT(*) as ref_count 
        FROM users 
        WHERE referrer_id IS NOT NULL 
        GROUP BY referrer_id 
        ORDER BY ref_count DESC 
        LIMIT 5
    ''')
    top_referrers = c.fetchall()
    
    text = (
        "📈 *АНАЛИТИКА ЧЁ ЗА GPT*\n\n"
        "📅 *Новые пользователи (7 дней):*\n"
    )
    
    for stat in reversed(daily_stats):
        text += f"• {stat}\n"
    
    text += "\n🏆 *Топ рефереров:*\n"
    for i, (ref_id, count) in enumerate(top_referrers, 1):
        text += f"{i}. ID {ref_id}: {count} приглашенных\n"
    
    await message.reply(text, parse_mode='Markdown')

# Словарь забаненных пользователей (в реальном проекте лучше хранить в БД)
banned_users = set()

@dp.message_handler(commands=['ban_user'])
async def ban_user_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        target_id = int(message.get_args())
    except Exception:
        await message.reply("Используй: /ban_user <user_id>")
        return

    user = get_user(target_id)
    if not user:
        await message.reply("Пользователь не найден.")
        return

    banned_users.add(target_id)
    await message.reply(f"🚫 Пользователь {target_id} заблокирован.", parse_mode='Markdown')

@dp.message_handler(commands=['unban_user'])
async def unban_user_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        target_id = int(message.get_args())
    except Exception:
        await message.reply("Используй: /unban_user <user_id>")
        return

    if target_id in banned_users:
        banned_users.remove(target_id)
        await message.reply(f"✅ Пользователь {target_id} разблокирован.", parse_mode='Markdown')
    else:
        await message.reply(f"Пользователь {target_id} не был заблокирован.", parse_mode='Markdown')

# Переменная для хранения цены подписки
SUBSCRIPTION_PRICE = 49
# --- Конфиг оплаты (без .env) ---
# Включить оплату звёздами Telegram (XTR)
USE_TELEGRAM_STARS = True
# Цена подписки в звёздах
SUBSCRIPTION_PRICE_STARS = 50

@dp.message_handler(commands=['set_price'])
async def set_price_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    try:
        new_price = int(message.get_args())
        if new_price < 0:
            raise ValueError("Цена не может быть отрицательной")
    except Exception:
        await message.reply("Используй: /set_price <цена в рублях>")
        return

    global SUBSCRIPTION_PRICE
    SUBSCRIPTION_PRICE = new_price
    await message.reply(f"💰 Цена подписки изменена на {new_price} рублей.", parse_mode='Markdown')

@dp.message_handler(commands=['logs'])
async def logs_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    # Получаем последние логи (можно настроить более детально)
    text = (
        "📋 *ПОСЛЕДНИЕ ЛОГИ*\n\n"
        "🕐 Время запуска: " + datetime.now().strftime('%d.%m.%Y %H:%M:%S') + "\n"
        "👥 Пользователей в памяти: " + str(len(user_histories)) + "\n"
        "🚫 Заблокированных: " + str(len(banned_users)) + "\n"
        "💰 Текущая цена: " + str(SUBSCRIPTION_PRICE) + " руб.\n\n"
        "📊 Для детальной статистики используй /stats"
    )
    await message.reply(text, parse_mode='Markdown')

@dp.message_handler(commands=['adminp'])
async def admin_panel_cmd(message: types.Message):
    if message.from_user.id not in ADMINS:
        await message.reply("⛔ У тебя нет прав для этой команды.")
        return
    
    text = (
        "🔧 *АДМИН ПАНЕЛЬ ЧЁ ЗА GPT*\n\n"
        "Выбери действие из меню ниже:"
    )
    
    # Создаем инлайн-клавиатуру
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton("📊 Статистика", callback_data="admin_stats"),
        InlineKeyboardButton("👥 Пользователи", callback_data="admin_users")
    )
    keyboard.add(
        InlineKeyboardButton("⏰ Выдать подписку", callback_data="admin_give_days"),
        InlineKeyboardButton("🔄 Сбросить пробник", callback_data="admin_reset_trial")
    )
    keyboard.add(
        InlineKeyboardButton("🗑️ Удалить пользователя", callback_data="admin_delete_user"),
        InlineKeyboardButton("📢 Рассылка", callback_data="admin_broadcast")
    )
    keyboard.add(
        InlineKeyboardButton("📁 Экспорт", callback_data="admin_export"),
        InlineKeyboardButton("📈 Аналитика", callback_data="admin_analytics")
    )
    keyboard.add(
        InlineKeyboardButton("🚫 Бан/Разбан", callback_data="admin_ban"),
        InlineKeyboardButton("💰 Цена", callback_data="admin_price")
    )
    keyboard.add(
        InlineKeyboardButton("📋 Логи", callback_data="admin_logs"),
        InlineKeyboardButton("❓ Справка", callback_data="admin_help")
    )
    
    await message.reply(text, reply_markup=keyboard, parse_mode='Markdown')

# Обработчики для инлайн-кнопок админ панели
@dp.callback_query_handler(lambda c: c.data.startswith('admin_'))
async def admin_callback_handler(callback_query: types.CallbackQuery):
    if callback_query.from_user.id not in ADMINS:
        await callback_query.answer("⛔ У тебя нет прав для этой команды.")
        return
    
    action = callback_query.data
    
    if action == "admin_stats":
        # Показываем статистику
        now = datetime.now()
        c.execute('SELECT COUNT(*) FROM users')
        total_users = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM users WHERE sub_until > ?', (now.isoformat(),))
        active_subs = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM users WHERE trial_until > ?', (now.isoformat(),))
        active_trials = c.fetchone()[0]
        
        text = (
            "📊 *СТАТИСТИКА ЧЁ ЗА GPT*\n\n"
            f"👥 *Всего пользователей:* {total_users}\n"
            f"💳 *Активных подписок:* {active_subs}\n"
            f"🎁 *Активных пробников:* {active_trials}\n"
            f"🚫 *Заблокированных:* {len(banned_users)}\n"
            f"💰 *Цена подписки:* {SUBSCRIPTION_PRICE} руб.\n\n"
            f"📈 *Конверсия:* {round((active_subs + active_trials) / total_users * 100, 1) if total_users > 0 else 0}%"
        )
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    elif action == "admin_users":
        # Показываем список пользователей (первые 10)
        c.execute('SELECT user_id, referrer_id, sub_until, trial_until FROM users ORDER BY user_id LIMIT 10')
        users = c.fetchall()
        
        text = "👥 *ПОСЛЕДНИЕ 10 ПОЛЬЗОВАТЕЛЕЙ:*\n\n"
        for user in users:
            uid, ref, sub, trial = user
            status = "🟢 Активен" if (sub and datetime.fromisoformat(sub) > datetime.now()) or (trial and datetime.fromisoformat(trial) > datetime.now()) else "🔴 Неактивен"
            text += f"ID: {uid} | {status}\n"
        
        text += "\n💡 Для полного списка используй /users"
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    elif action == "admin_analytics":
        # Показываем аналитику
        now = datetime.now()
        c.execute('SELECT COUNT(*) FROM users WHERE trial_until > ?', ((now - timedelta(days=7)).isoformat(),))
        new_users_week = c.fetchone()[0]
        
        c.execute('''
            SELECT referrer_id, COUNT(*) as ref_count 
            FROM users 
            WHERE referrer_id IS NOT NULL 
            GROUP BY referrer_id 
            ORDER BY ref_count DESC 
            LIMIT 3
        ''')
        top_referrers = c.fetchall()
        
        text = (
            "📈 *АНАЛИТИКА ЧЁ ЗА GPT*\n\n"
            f"🆕 *Новых за неделю:* {new_users_week}\n\n"
            "🏆 *Топ рефереров:*\n"
        )
        
        for i, (ref_id, count) in enumerate(top_referrers, 1):
            text += f"{i}. ID {ref_id}: {count} приглашенных\n"
        
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    elif action == "admin_logs":
        # Показываем логи
        text = (
            "📋 *СИСТЕМНЫЕ ЛОГИ*\n\n"
            f"🕐 Время: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n"
            f"👥 В памяти: {len(user_histories)} пользователей\n"
            f"🚫 Заблокировано: {len(banned_users)}\n"
            f"💰 Цена: {SUBSCRIPTION_PRICE} руб.\n"
            f"💾 База данных: users.db\n\n"
            "✅ Система работает стабильно"
        )
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    elif action == "admin_help":
        # Показываем справку
        text = (
            "❓ *СПРАВКА ПО АДМИН ПАНЕЛИ*\n\n"
            "*Команды:*\n"
            "• `/adminp` - Открыть админ панель\n"
            "• `/stats` - Детальная статистика\n"
            "• `/users` - Список всех пользователей\n"
            "• `/give_days <id> <дни>` - Выдать подписку\n"
            "• `/reset_trial <id>` - Сбросить пробник\n"
            "• `/delete_user <id>` - Удалить пользователя\n"
            "• `/broadcast <текст>` - Рассылка\n"
            "• `/export_users` - Экспорт данных\n"
            "• `/analytics` - Аналитика\n"
            "• `/ban_user <id>` - Забанить\n"
            "• `/unban_user <id>` - Разбанить\n"
            "• `/set_price <цена>` - Изменить цену\n"
            "• `/logs` - Системные логи\n\n"
            "💡 Используй кнопки выше для быстрого доступа!"
        )
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    elif action in ["admin_give_days", "admin_reset_trial", "admin_delete_user", "admin_broadcast", "admin_export", "admin_ban", "admin_price"]:
        # Для этих действий нужен дополнительный ввод
        text = (
            f"🔧 *{action.replace('admin_', '').upper()}*\n\n"
            "Эта функция требует дополнительных параметров.\n"
            "Используй соответствующие команды:\n\n"
        )
        
        if action == "admin_give_days":
            text += "`/give_days <user_id> <дни>`\nПример: `/give_days 123456789 30`"
        elif action == "admin_reset_trial":
            text += "`/reset_trial <user_id>`\nПример: `/reset_trial 123456789`"
        elif action == "admin_delete_user":
            text += "`/delete_user <user_id>`\nПример: `/delete_user 123456789`"
        elif action == "admin_broadcast":
            text += "`/broadcast <текст>`\nПример: `/broadcast Привет всем! 🚀`"
        elif action == "admin_export":
            text += "`/export_users` - экспортирует всех пользователей в CSV"
        elif action == "admin_ban":
            text += "`/ban_user <user_id>` - заблокировать\n`/unban_user <user_id>` - разблокировать"
        elif action == "admin_price":
            text += "`/set_price <цена>`\nПример: `/set_price 50`"
        
        await callback_query.message.edit_text(text, parse_mode='Markdown')
    
    await callback_query.answer()

# --- Обработка кнопок ---
@dp.message_handler(lambda m: m.text == '📊 Статус')
async def status_btn(message: types.Message):
    user = get_user(message.from_user.id)
    now = datetime.now()
    sub_until = datetime.fromisoformat(user[2]) if user and user[2] else None
    trial_until = datetime.fromisoformat(user[3]) if user and user[3] else None
    if (sub_until and sub_until > now) or (trial_until and trial_until > now):
        # Есть активный доступ
        until = sub_until if (sub_until and (not trial_until or sub_until > trial_until)) else trial_until
        left = until - now
        days = left.days
        hours = left.seconds // 3600
        text = (
            "📊 *ЧЁ ЗА GPT: Твой статус*\n\n"
            "✅ *Доступ активен!*\n\n"
            f"⏱️ *Осталось времени:* {days} дней {hours} часов (до {until.strftime('%d.%m.%Y %H:%M')})\n\n"
            "👉 Пользуйся на полную! Пиши запросы – нейросеть ждет!"
        )
    else:
        text = (
            "📊 *ЧЁ ЗА GPT: Твой статус*\n\n"
            "⛔ *Доступ закрыт!*\n\n"
            "Твой пробный период / подписка закончились.\n\n"
            "🔥 Не останавливайся! Нажми *'ОПЛАТА'* и продолжай получать четкие ответы без границ!\n"
            "👉 Или *'РЕФЕРАЛЫ'* – пригласи друга и получите по +3 дня бесплатно!"
        )
    await message.answer(text, parse_mode='Markdown')

@dp.message_handler(lambda m: m.text == '👥 Рефералы')
async def referrals_btn(message: types.Message):
    user_id = message.from_user.id
    bot_info = await bot.get_me()
    bot_username = bot_info.username
    ref_link = f"https://t.me/{bot_username}?start={user_id}"
    c.execute('SELECT COUNT(*) FROM users WHERE referrer_id=?', (user_id,))
    count = c.fetchone()[0]
    bonus_days = count * 3  # +3 дня за каждого друга
    text = (
        "🤝 *ЧЁ ЗА GPT: Зови друзей – получай халяву!*\n\n"
        f"Есть друг, которому нужен четкий AI? Делись своей рефкой:\n\n"
        f"`{ref_link}`\n\n"
        "*🔥 ЗА КАЖДОГО ДРУГА ПО ТВОЕЙ ССЫЛКЕ:*\n"
        "1. *ОН* получает свои ⏱️ 3 БЕСПЛАТНЫХ ДНЯ!\n"
        "2. *ТЫ* получаешь +⏱️ 1 ДЕНЬ к *своей* подписке или пробнику! 💥\n\n"
        f"Ты уже пригласил: *{count}* друзей (+{bonus_days} дней)\n\n"
        "👉 Кидай ссылку в чаты, соцсети – пусть народ валит! Больше друзей = больше твоего бесплатного времени с ЧЁ ЗА GPT!"
    )
    await message.answer(text, parse_mode='Markdown')

@dp.message_handler(lambda m: m.text == '💳 Оплата')
async def pay_btn(message: types.Message):
    if USE_TELEGRAM_STARS:
        text = (
            "🔥 ЧЁ ЗА GPT — твой дерзкий AI-братан!\n\n"
            f"💥 *Подписка за {SUBSCRIPTION_PRICE_STARS} ⭐️ в месяц!*\n"
            "Оплата через Telegram Stars (внутренняя валюта).\n\n"
            "Что ты получаешь?\n"
            "✅ Неограниченные запросы\n"
            "✅ Быстрые ответы 24/7\n"
            "✅ Голосовые и текстовые ответы\n"
            "✅ Доступ ко всем фишкам бота\n"
            "✅ Пробный период 3 дня для новых пользователей\n"
            "✅ Поддержка и обновления\n\n"
            "💡 *Оплати прямо в Telegram Stars и пользуйся без ограничений!*"
        )
        prices = [LabeledPrice(label="Подписка на месяц", amount=int(SUBSCRIPTION_PRICE_STARS))]
        await bot.send_invoice(
            message.chat.id,
            title="Подписка на ЧЁ ЗА GPT",
            description=text,
            provider_token="",
            currency="XTR",
            prices=prices,
            start_parameter="gpt-subscription-stars",
            payload=str(message.from_user.id)
        )
    else:
        text = (
            "🔥 ЧЁ ЗА GPT — твой дерзкий AI-братан!\n\n"
            f"💥 *Подписка всего за {SUBSCRIPTION_PRICE} ₽/месяц!*\n"
            "Что ты получаешь?\n"
            "✅ Неограниченные запросы\n"
            "✅ Быстрые ответы 24/7\n"
            "✅ Голосовые и текстовые ответы\n"
            "✅ Доступ ко всем фишкам бота\n"
            "✅ Пробный период 3 дня для новых пользователей\n"
            "✅ Поддержка и обновления\n\n"
            "💡 *Оплати прямо в Telegram и пользуйся без ограничений!*"
        )
        provider_token = os.getenv("TELEGRAM_PROVIDER_TOKEN", "")
        amount_kopecks = int(SUBSCRIPTION_PRICE) * 100
        prices = [LabeledPrice(label="Подписка на месяц", amount=amount_kopecks)]
        if not provider_token:
            await message.answer("Оплата временно недоступна: не настроен провайдер. Свяжись с админом.", parse_mode='Markdown')
            return
        await bot.send_invoice(
            message.chat.id,
            title="Подписка на ЧЁ ЗА GPT",
            description=text,
            provider_token=provider_token,
            currency="RUB",
            prices=prices,
            start_parameter="gpt-subscription",
            payload=str(message.from_user.id)
        )

@dp.pre_checkout_query_handler(lambda query: True)
async def pre_checkout_query(pre_checkout_q: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(pre_checkout_q.id, ok=True)

@dp.message_handler(content_types=types.ContentType.SUCCESSFUL_PAYMENT)
async def successful_payment(message: types.Message):
    try:
        user_id = int(message.successful_payment.invoice_payload)
    except Exception:
        user_id = message.from_user.id
    # Продлим/установим подписку на +30 дней
    now = datetime.now()
    user = get_user(user_id)
    current_until = datetime.fromisoformat(user[2]) if user and user[2] else now
    new_until = max(current_until, now) + timedelta(days=30)
    c.execute('UPDATE users SET sub_until=? WHERE user_id=?', (new_until.isoformat(), user_id))
    conn.commit()
    await message.answer(f"✅ Оплата прошла успешно! Подписка активна до {new_until.strftime('%d.%m.%Y %H:%M')}.")

@dp.message_handler(lambda m: m.text == 'ℹ️ О БОТЕ')
async def about_bot_btn(message: types.Message):
    text = (
        "🚀 *ЧЁ ЗА GPT?*\n\n"
        "Не бот, а дерзкий братан в Телеге.\n"
        "Забудь занудные нейросети – тут перец с огнём! 🌶🔥\n\n"
        "💪 *ЧЁ МОЖЕТ?*\n\n"
        "🎙️ *Озвучка:* кидаешь текст или войс – получаешь ответ голосом, будто мы на созвоне.\n\n"
        "⚡️ *Прокачка промта:* написал \"диета\" или \"смайл\"? Я сам превращу в жирный запрос и дам толковый ответ.\n\n"
        "🤟 *Реальное общение:* прикалываюсь, рофлю, но всегда по делу.\n\n"
        "🤖 *ФИШКИ ПЕРЦА*\n\n"
        "Мозги GPT-4 Turbo 🧠\n\n"
        "Реакция быстрее, чем ты моргнёшь ⚡️\n\n"
        "Харизма: 100% живого общения, 0% унылого бота 😎\n\n"
        "🌶 *ЗАЧЕМ ТЕБЕ Я?*\n\n"
        "Курсач завтра – я спасу ночь 📚\n\n"
        "Надо разъебать препода шуткой – подскажу 😈\n\n"
        "Мем для чата – сделаю 🔥\n\n"
        "Даже с \"теми самыми\" темами поговорим без лишнего 🤫\n\n"
        "👇 Хочешь кайфовать – жми */start* и нюхни этот перец! 👇"
    )
    await message.answer(text, parse_mode='Markdown')

@dp.message_handler(lambda m: m.text == '🖼 Картинка')
async def picture_btn(message: types.Message):
    waiting_for_image_prompt.add(message.from_user.id)
    await message.answer('Опиши, что нарисовать (например: "кот гоняется за мышкой реалистично") — и я сгенерирую картинку!', parse_mode='Markdown')

# --- Основной обработчик сообщений (нейросеть) ---
@dp.message_handler()
async def gpt_stub(message: types.Message):
    if message.from_user.id in waiting_for_image_prompt:
        waiting_for_image_prompt.remove(message.from_user.id)
        prompt = message.text.strip()
        if not prompt:
            await message.reply('Пожалуйста, опиши, что нарисовать!')
            return
        await message.reply('🎨 Ща наколдую тебе картинку, держись! Это может занять до минуты...')
        try:
            img_url = await neuroimg_generate(prompt)
            await message.reply_photo(img_url, caption=f'Твоя картинка по запросу: {prompt}')
        except Exception as e:
            await message.reply(f'[Ошибка] Не удалось сгенерировать картинку: {e}')
        return
    if message.text in ['📊 Статус', '👥 Рефералы', '💳 Оплата', 'ℹ️ О БОТЕ']:
        return
    if message.from_user.id in banned_users:
        await message.answer("🚫 Ты заблокирован администратором.", parse_mode='Markdown')
        return
    user = get_user(message.from_user.id)
    now = datetime.now()
    if not user:
        await message.answer("Сначала напиши /start")
        return
    sub_until = datetime.fromisoformat(user[2]) if user[2] else None
    trial_until = datetime.fromisoformat(user[3]) if user[3] else None
    model_patterns = [
        r'какая (у тебя|ты|вы) модель', r'что за модель', r'какой ты (бот|модель)', r'what model', r'who are you',
        r'which model', r'что ты за (бот|модель)', r'какая версия', r'are you gpt', r'are you chatgpt', r'are you openai',
        r'какой ты gpt', r'какой gpt', r'какая gpt', r'какая нейросеть', r'what ai', r'what neural', r'what version',
        r'who made you', r'who created you', r'who is your creator', r'кто тебя создал', r'кто твой создатель', r'кто ты',
    ]
    user_text = message.text.lower()
    if any(re.search(p, user_text) for p in model_patterns):
        await message.answer("Я бот ЧЁ ЗА GPT", parse_mode='Markdown')
        return
    if (sub_until and sub_until > now) or (trial_until and trial_until > now):
        system_prompt = SYSTEM_PROMPT
        user_id = message.from_user.id
        if user_id not in user_histories:
            user_histories[user_id] = []
        user_histories[user_id].append({"role": "user", "content": message.text})
        user_histories[user_id] = user_histories[user_id][-10:]
        history = user_histories[user_id].copy()

        research_result = ""
        if needs_research(message.text):
            await message.answer("🔎 Ищу инфу в интернете...")
            research_result = langsearch_query(message.text)
            await message.answer(f"[DEBUG] LangSearch ответ:\n{research_result}")
            if research_result:
                research_block = f"Вот что удалось найти в интернете (LangSearch):\n{research_result}\n\n"
            else:
                research_block = ""
        else:
            research_block = ""

        messages_to_send = [
            {"role": "system", "content": research_block + system_prompt}
        ] + history
        import openai
        try:
            client = openai.OpenAI(
                api_key="sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg=",
                base_url="https://router.requesty.ai/v1",
                default_headers={"Authorization": f"Bearer sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg="}
            )
            response = client.chat.completions.create(
                model="azure/openai/gpt-4.1",
                messages=messages_to_send
            )
            if not response.choices:
                await message.answer("[Ошибка] Нейросеть не дала ответа.", parse_mode='Markdown')
                return
            assistant_answer = response.choices[0].message.content
            user_histories[user_id].append({"role": "assistant", "content": assistant_answer})
            user_histories[user_id] = user_histories[user_id][-10:]
            await send_long_message(message, assistant_answer)
        except openai.OpenAIError as e:
            await message.answer(f"[Ошибка OpenAI API] {e}", parse_mode='Markdown')
        except Exception as e:
            await message.answer(f"[Ошибка] {e}", parse_mode='Markdown')
    else:
        await message.answer("У тебя закончился пробный период и нет активной подписки. Оформи подписку через /pay", parse_mode='Markdown')

# --- Голосовой обработчик ---
@dp.message_handler(content_types=[types.ContentType.VOICE])
async def handle_voice(message: types.Message):
    user_id = message.from_user.id
    user = get_user(user_id)
    now = datetime.now()
    if not user:
        await message.answer("Сначала напиши /start")
        return
    sub_until = datetime.fromisoformat(user[2]) if user[2] else None
    trial_until = datetime.fromisoformat(user[3]) if user[3] else None
    if not ((sub_until and sub_until > now) or (trial_until and trial_until > now)):
        await message.answer("У тебя закончился пробный период и нет активной подписки. Оформи подписку через /pay", parse_mode='Markdown')
        return
    ogg_filename = f"voice_{user_id}.ogg"
    wav_filename = f"voice_{user_id}.wav"
    response_filename = f"response_{user_id}.mp3"
    try:
        voice_file_info = await bot.get_file(message.voice.file_id)
        voice_ogg = await bot.download_file(voice_file_info.file_path)
        with open(ogg_filename, "wb") as f:
            f.write(voice_ogg.read())
        try:
            audio = AudioSegment.from_ogg(ogg_filename)
            audio.export(wav_filename, format="wav")
        except Exception as conv_err:
            logging.error(f"pydub OGG->WAV error: {conv_err}. Trying ffmpeg fallback...")
            # Fallback через ffmpeg CLI
            import subprocess
            subprocess.run([ffmpeg_path or 'ffmpeg', '-y', '-i', ogg_filename, wav_filename], check=True)
        r = sr.Recognizer()
        user_text = ""
        with sr.AudioFile(wav_filename) as source:
            audio_data = r.record(source)
            try:
                user_text = r.recognize_google(audio_data, language="ru-RU")
            except Exception as rec_err:
                logging.error(f"SpeechRecognition error: {rec_err}")
                # Если распознавание упало — всё равно продолжим и озвучим базовый ответ
                user_text = "Озвучь ответ на мой голос."
        if user_id not in user_histories:
            user_histories[user_id] = []
        user_histories[user_id].append({"role": "user", "content": user_text})
        user_histories[user_id] = user_histories[user_id][-10:]
        history = user_histories[user_id].copy()
        system_prompt = SYSTEM_PROMPT
        messages_to_send = [{"role": "system", "content": system_prompt}] + history
        import openai
        client = openai.OpenAI(
            api_key="sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg=",
            base_url="https://router.requesty.ai/v1",
            default_headers={"Authorization": f"Bearer sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg="}
        )
        response = client.chat.completions.create(
            model="azure/openai/gpt-4.1",
            messages=messages_to_send
        )
        assistant_answer = response.choices[0].message.content
        user_histories[user_id].append({"role": "assistant", "content": assistant_answer})
        user_histories[user_id] = user_histories[user_id][-10:]
        # Очищаем текст для озвучки: убираем эмодзи, markdown, фирменные фразы
        text_for_speech = assistant_answer
        # Удаляем эмодзи
        text_for_speech = re.sub(r'[\U00010000-\U0010ffff]', '', text_for_speech)
        # Удаляем markdown-символы
        text_for_speech = re.sub(r'[\*\_\~\`\>\#\-\=\[\]\(\)]', '', text_for_speech)
        # Удаляем фирменные фразы (если появятся новые)
        # text_for_speech = re.sub(r'какая-то фраза.*?\n', '', text_for_speech, flags=re.IGNORECASE)
        text_for_speech = text_for_speech.strip()
        if not text_for_speech:
            text_for_speech = "Голосовой ответ."
        try:
            if TTS_ENGINE == 'edge':
                communicate = edge_tts.Communicate(text_for_speech, EDGE_VOICE, rate=EDGE_RATE, volume=EDGE_VOLUME)
                await communicate.save(response_filename)
            else:
                tts = gTTS(text=text_for_speech, lang=VOICE_LANG, tld=VOICE_TLD, slow=VOICE_SLOW)
                tts.save(response_filename)
        except Exception as tts_err:
            logging.error(f"TTS error: {tts_err}")
            if TTS_FALLBACK_TO_GTTS:
                tts = gTTS(text=text_for_speech, lang=VOICE_LANG, tld=VOICE_TLD, slow=VOICE_SLOW)
                tts.save(response_filename)
            else:
                raise
        with open(response_filename, "rb") as audio_response_file:
            await message.answer_voice(audio_response_file, parse_mode='Markdown')
    except sr.UnknownValueError as e:
        logging.exception("SpeechRecognition UnknownValueError")
        await message.answer(f"Не расслышал речь ({e.__class__.__name__}). Скажи ещё раз.", parse_mode='Markdown')
    except sr.RequestError as e:
        logging.exception("SpeechRecognition RequestError")
        await message.answer(f"Проблема с сервисом распознавания: {e}", parse_mode='Markdown')
    except Exception as e:
        logging.exception("Voice handler error")
        await message.answer(f"Ошибка обработки голоса: {e.__class__.__name__}: {e}", parse_mode='Markdown')
    finally:
        if os.path.exists(ogg_filename): os.remove(ogg_filename)
        if os.path.exists(wav_filename): os.remove(wav_filename)
        if os.path.exists(response_filename): os.remove(response_filename)

# --- Команда для теста TTS без распознавания ---
@dp.message_handler(commands=['say'])
async def say_cmd(message: types.Message):
    text = message.get_args() or "Привет! Это тест озвучки в боте."
    response_filename = f"say_{message.from_user.id}.mp3"
    try:
        to_say = text.strip() or "Тест озвучки."
        try:
            if TTS_ENGINE == 'edge':
                communicate = edge_tts.Communicate(to_say, EDGE_VOICE, rate=EDGE_RATE, volume=EDGE_VOLUME)
                await communicate.save(response_filename)
            else:
                tts = gTTS(text=to_say, lang=VOICE_LANG, tld=VOICE_TLD, slow=VOICE_SLOW)
                tts.save(response_filename)
        except Exception as tts_err:
            logging.error(f"/say TTS error: {tts_err}")
            if TTS_FALLBACK_TO_GTTS:
                tts = gTTS(text=to_say, lang=VOICE_LANG, tld=VOICE_TLD, slow=VOICE_SLOW)
                tts.save(response_filename)
            else:
                await message.answer(f"TTS ошибка: {tts_err}")
                return
        with open(response_filename, "rb") as f:
            await message.answer_voice(f)
    except Exception as e:
        logging.error(f"/say error: {e}")
        await message.answer("Не удалось озвучить текст.")
    finally:
        if os.path.exists(response_filename):
            os.remove(response_filename)

# --- Обработчик фото (нейросеть) ---
@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    user = get_user(user_id)
    now = datetime.now()
    if not user:
        await message.answer("Сначала напиши /start")
        return
    sub_until = datetime.fromisoformat(user[2]) if user[2] else None
    trial_until = datetime.fromisoformat(user[3]) if user[3] else None
    if not ((sub_until and sub_until > now) or (trial_until and trial_until > now)):
        await message.answer("У тебя закончился пробный период и нет активной подписки. Оформи подписку через /pay", parse_mode='Markdown')
        return
        # --- Системный промт ---
    system_prompt = SYSTEM_PROMPT
    # Формируем контент для сообщения (текст + фото)
    user_text = message.caption if message.caption else "Что на этой картинке?"
    # Получаем фото
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file = await bot.download_file(file_info.file_path)
    import base64
    data_url = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode()
    if user_id not in user_histories:
        user_histories[user_id] = []
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    user_histories[user_id].append({"role": "user", "content": content})
    user_histories[user_id] = user_histories[user_id][-10:]
    history = user_histories[user_id].copy()
    messages_to_send = [{"role": "system", "content": system_prompt}] + history
    import openai
    try:
        client = openai.OpenAI(
            api_key="sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg=",
            base_url="https://router.requesty.ai/v1",
            default_headers={"Authorization": f"Bearer sk-5ewuexmtSvmrFplLVAntfbGtPyv0uBvjlYyGyPWVunyW2eN540b2a6eP2tZ0uODUjWmE/hI5Z6o5lQZgA14J4a8WGsIjLyJ3AxPD58pO+wg="}
        )
        response = client.chat.completions.create(
            model="azure/openai/gpt-4.1",
            messages=messages_to_send,
            max_tokens=2048
        )
        assistant_answer = response.choices[0].message.content
        user_histories[user_id].append({"role": "assistant", "content": assistant_answer})
        user_histories[user_id] = user_histories[user_id][-10:]
        await send_long_message(message, assistant_answer)
    except Exception as e:
        await message.answer(f"Что-то пошло не так при обработке фото: {e}", parse_mode='Markdown')
        logging.error(f"Photo processing error: {e}")

@dp.message_handler(commands=['draw'])
async def draw_cmd(message: types.Message):
    prompt = message.get_args() or message.text if message.text != '/draw' else ''
    if not prompt or prompt.strip() == '/draw':
        await message.reply("Используй: /draw <описание картинки>\n\nПример: /draw кот на скейтборде в стиле аниме")
        return
    await message.reply("🎨 Генерирую картинку, подожди (это может занять до минуты)...")
    try:
        img_url = await neuroimg_generate(prompt)
        await message.reply_photo(img_url, caption=f"Твоя картинка по запросу: {prompt}")
    except Exception as e:
        await message.reply(f"[Ошибка] Не удалось сгенерировать картинку: {e}")

# --- Ожидание промпта для генерации картинки ---
waiting_for_image_prompt = set()

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True) 