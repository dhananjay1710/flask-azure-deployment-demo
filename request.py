import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'GRE Score':330, 'TOEFL Score':110, 'CGPA':8.5})

print(r.json())