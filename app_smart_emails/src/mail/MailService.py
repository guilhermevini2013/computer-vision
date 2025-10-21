import imaplib
import re
import threading
import email

def verify_account(user, password):
    print("Authenticando...")
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(user, password)
        return True
    except:
        return False

class MailService:
    def __init__(self, modelAIService, user, password):
        self.modelAIService = modelAIService
        self.IMAP_HOST = "imap.gmail.com"
        self.IMAP_USER = user
        self.IMAP_PASS = password
        self.mail = self.login_connection_mail()
        self._create_labels_once()

        self._stop_event = threading.Event()
        self._thread = None

    def login_connection_mail(self):
        mail = imaplib.IMAP4_SSL(self.IMAP_HOST)
        mail.login(self.IMAP_USER, self.IMAP_PASS)
        return mail

    def logout_connection_mail(self):
        try:
            self.mail.logout()
        except Exception:
            pass

    def _create_labels_once(self):
        try:
            self.mail.create('"100% Phishing"')
            self.mail.create('"75% Phishing"')
            self.mail.create('"50% Phishing"')
            self.mail.create('"25% Phishing"')
            self.mail.create('"Safe"')
        except Exception:
            pass

    def count_urls(self,text):
        url_pattern = r'(https?://\S+|www\.\S+)'
        urls = re.findall(url_pattern, text)
        return len(urls)

    def check_mails(self):
        delay_to_wait = 0
        while not self._stop_event.is_set():
            self.mail.select('"[Gmail]/All Mail"')
            status, data = self.mail.search(None, "UNSEEN")
            email_ids = data[0].split()

            if not email_ids:
                delay_to_wait = 0
            else:
                for num in email_ids:

                    status, msg_data = self.mail.fetch(num, "(RFC822)")
                    if status != "OK":
                        print(f"Falha ao buscar {num.decode()}")
                        continue

                    raw_email = msg_data[0][1]
                    try:
                        msg = email.message_from_bytes(raw_email)
                    except:
                        continue

                    # Subject
                    subject = email.header.decode_header(msg["Subject"])[0]
                    if isinstance(subject[0], bytes):
                        subject = subject[0].decode(subject[1] if subject[1] else "utf-8")
                    else:
                        subject = subject[0]

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_dispo = str(part.get("Content-Disposition"))

                            if content_type == "text/plain" and "attachment" not in content_dispo:
                                body_bytes = part.get_payload(decode=True)
                                charset = part.get_content_charset()
                                body = body_bytes.decode(charset if charset else "utf-8")
                                break
                    else:
                        body_bytes = msg.get_payload(decode=True)
                        charset = msg.get_content_charset()
                        body = body_bytes.decode(charset if charset else "utf-8")

                    urls = self.count_urls(body)
                    data_check = {
                        "subject": subject,
                        "body": body,
                        "urls": urls
                    }
                    pred = self.modelAIService.predict(data_check)
                    if 0.0 <= pred <= 0.1:
                        self.mail.copy(num, '"Safe"')
                    elif 0.1 < pred <= 0.4:  # Começa acima de 0.1 para evitar sobreposição
                        self.mail.copy(num, '"25% Phishing"')
                    elif 0.4 < pred <= 0.6:  # Começa acima de 0.4
                        self.mail.copy(num, '"50% Phishing"')
                    elif 0.6 < pred <= 0.85:  # Começa acima de 0.6
                        self.mail.copy(num, '"75% Phishing"')
                    elif 0.85 < pred <= 1.0:  # Começa acima de 0.85
                        self.mail.copy(num, '"100% Phishing"')

            self._stop_event.wait(delay_to_wait)

        self.logout_connection_mail()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.check_mails, daemon=True)
        self._thread.start()

    def stop(self, timeout=5):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                print("Thread não terminou no tempo limite.")
            else:
                print("Serviço parado.")

