import requests

url = 'http://localhost:5000/predict_api'
r = request.post(url,json('pregnancies': 23, 'glucose', 'bloodpressure':56, 'skinthickness':45, 'insulin':67, 'bmi':76, 'diabetespedigreefunction':78, 'age':67))

print(r.json())