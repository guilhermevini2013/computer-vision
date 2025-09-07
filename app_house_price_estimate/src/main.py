from app_house_price_estimate.src.ann.network import Network
from app_house_price_estimate.src.csvService import CsvService
from app_house_price_estimate.src.scraping import Scraping
from sklearn.preprocessing import MinMaxScaler

all_houses = []
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

def start_scraping():
    scraping = Scraping()
    houses = scraping.get_information()
    all_houses.extend(houses)

csv_service = CsvService(scaler_X, scaler_y)
network_service = Network(csv_service, scaler_X, scaler_y)

### data already retrieved from scraping.
#start_scraping()

#X, y = csv_service.normalize_data("./houses_data.csv")

#network_service.create_train_network(X, y)

new_house_data = {
    'area': 160,
    'bedroom': 3,
    'bathroom': 2,
    'garage': 2,
    'neighborhood': 'jardim aurelia'
}

predict = network_service.predict_new_house(new_house_data)
print("Detalhes da casa para previsão:")
print(f"Área: {new_house_data['area']} m²")
print(f"Quartos: {new_house_data['bedroom']}")
print(f"Banheiros: {new_house_data['bathroom']}")
print(f"Garagem: {new_house_data['garage']} vaga(s)")
print(f"Bairro: {new_house_data['neighborhood'].title()}")

print("\nPreço estimado da casa:")
print(f"R$ {predict}")

