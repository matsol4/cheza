import telebot

print("Стартуем!")

try:
    bot = telebot.TeleBot("7652536635:AAEpiCQecQbc-tP2Tm_6_BCNetZa8jz2JhY")

    @bot.message_handler(content_types=['text'])
    def eho(message):
        print("Получено сообщение:", message.text)
        bot.send_message(message.chat.id, message.text)

    print("Перед polling")
    bot.polling(none_stop=True)
    print("После polling")
except Exception as e:
    print("ОШИБКА:", e) 