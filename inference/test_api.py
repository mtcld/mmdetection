import requests

def predict(uuid):
    print('Predicting here')
    url = "http://127.0.0.1:8080/predictions/" + "AI_api"
    data = {'uuid':uuid,'vin':'None'}
    r = requests.post(url, data=data)
    print('r')
    print(r)
    if r.ok:
        print('requests is ok')
        j = r.json()
        print(j)

uuid='1abc'
predict(uuid)
