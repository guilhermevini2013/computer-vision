import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

from app_house_price_estimate.src.csvService import CsvService
from app_house_price_estimate.src.network import Network
from app_house_price_estimate.src.scraping import Scraping

app = Flask(__name__)
CORS(app)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

csv_service = CsvService(scaler_X, scaler_y)
network_service = Network(csv_service, scaler_X, scaler_y)
scrapping = Scraping()

def scraping_train_network():
    all_houses = []

    def start_scraping():
        scraping = Scraping()
        houses = scraping.get_information()
        all_houses.extend(houses)
    ### data already retrieved from scraping.
    #start_scraping()

    X, y = csv_service.normalize_data("./houses_data.csv")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.3, random_state=1)
    network_service.create_train_network(X_train, X_test, y_train, y_test)

@app.route('/neighborhoods', methods=["GET"])
def get_all_neighborhoods():
    all_neighborhoods = [n.split("/")[-2].replace("-", " ") for n in scrapping.get_neighborhood()]
    all_neighborhoods.sort()
    json_response = {
        "all_neighborhoods": all_neighborhoods
    }
    return jsonify(json_response)

@app.route('/preview-price', methods=["POST"])
def estimate_price_house():
    estimate_house = request.get_json()
    return jsonify({
        "estimated_price":network_service.predict_new_house(estimate_house)
    })

if __name__ == '__main__':
    scraping_train_network()
    app.run(debug=False)