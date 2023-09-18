N=50
locust -f test/locust/locust.py \
  --headless -u ${N} -r ${N} -i ${N} \
  --host=http://0.0.0.0:8080 \
  --html=test/~report.html