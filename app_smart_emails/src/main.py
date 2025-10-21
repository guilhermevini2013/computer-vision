import time

from app_smart_emails.src.mail.MailService import MailService, verify_account
from app_smart_emails.src.modelAI.TrainModelAI import TrainModelAi
from app_smart_emails.src.pandas.MailDataService import MailDataService

if __name__ == '__main__':
    modelAIService = TrainModelAi()
    banner = """
         ███████╗ █████╗ ███████╗███████╗              ███████╗███╗   ███╗ █████╗ ██╗██╗         
         ██╔════╝██╔══██╗██╔════╝██╔════╝              ██╔════╝████╗ ████║██╔══██╗██║██║         
         ███████╗███████║█████╗  █████╗      █████╗    █████╗  ██╔████╔██║███████║██║██║         
         ╚════██║██╔══██║██╔══╝  ██╔══╝      ╚════╝    ██╔══╝  ██║╚██╔╝██║██╔══██║██║██║         
         ███████║██║  ██║██║     ███████╗              ███████╗██║ ╚═╝ ██║██║  ██║██║███████╗    
         ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝              ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚══════╝    
                                                                                        
                                                                                        
               P H I S H I N G   D E F E N S E    |    P H I S H I N G   D E F E N S E
        """

    print("\n" * 1)
    print(banner)
    print("=" * 60)
    print("\n[ OPÇÕES ]")
    print("  1 - Iniciar Login e Monitoramento de E-mails (Monitoramento)")
    print("  2 - Iniciar Treinamento da Rede Neural (CNN)")

    option = input("Digite sua opção (1 ou 2): ").strip()
    if option == '2':
        mail_data_service = MailDataService()
        mail_data = mail_data_service.get_all_mail_train()
        modelAIService.train_model(mail_data)

    email_app = input("Digite seu email:")
    password_app = input("Digite sua senha do app:")
    while not verify_account(email_app, password_app):
        print("\n")
        print("Email ou senha app incorreto.")
        print("=" * 60)
        email_app = input("Digite seu email:")
        password_app = input("Digite sua senha do app:")

    mail_service = MailService(modelAIService, email_app, password_app)
    mail_service.start()
    print("=" * 60)

    print("=" * 60)
    print(" " * 10 + "🛡️ Sistema de Defesa Anti-Phishing Iniciado 🛡️")
    print("=" * 60)

    # === STATUS E INFORMAÇÕES ===
    print("\n[ STATUS INICIAL ]")
    print(f"  - Horário: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Serviço: {'ATIVO e MONITORANDO (Modo Polling)'}")
    print("\n" + "-" * 60)
    # === LEGENDA DE CLASSIFICAÇÃO ===
    print("CLASSIFICAÇÃO DE RISCO:")
    print("  ✅ SAFE: 0.0 - 0.1")
    print("  ⚠️ 25% Phishing: 0.1 - 0.4")
    print("  🚨 50% Phishing: 0.4 - 0.6")
    print("  🔥 75% Phishing: 0.6 - 0.85")
    print("  🛑 100% Phishing: 0.85 - 1.0")
    print("-" * 60)

    # === INSTRUÇÃO ===
    print("\n[ LOG ] Verificando caixa de entrada... Aguardando novos eventos...")
    print("============================================================\n")

    try:
        while True:
            continue
    except KeyboardInterrupt:
        mail_service.stop()
