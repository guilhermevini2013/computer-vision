import os
import pandas as pd

class MailDataService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def get_all_mail_train(self):
        path = os.path.join(self.base_dir, "dataset.csv")
        df = pd.read_csv(path, usecols=["subject", "body", "label", "urls"])

        mails = df.to_dict(orient="records")

        path = os.path.join(self.base_dir, "data2.csv")
        df = pd.read_csv(path, usecols=["subject", "body", "label", "urls"])

        mails.extend(df.to_dict(orient="records"))
        return mails
