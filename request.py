import requests

url = 'http://localhost:5000/predict_api'
r= requests.post(url,json={'step':743 , 'type':'CASH_OUT' , 'amount':339682 , 'nameOrig':'C786484425', 'oldbalanceOrg':339682 , 'newbalanceOrig':0 , 'nameDest':'C776919290' , 'oldbalanceDest':0 , 'newbalanceDest':339682 , 'isFlaggedFraud':0 })

print(r.json())