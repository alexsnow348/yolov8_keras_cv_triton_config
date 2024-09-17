curl -o /dev/null -s -w "\nTime taken: %{time_total} seconds\nDNS lookup time: %{time_namelookup} seconds\nConnect time: %{time_connect} seconds\nStart transfer time: %{time_starttransfer} seconds\n" -X POST http://localhost:8000/v2/models/ensemble_model/versions/1/infer      -H "Content-Type: application/json"  -d @input_data_fastercnn.json

curl  -o /dev/null -s -w "\nTime taken: %{time_total} seconds\nDNS lookup time: %{time_namelookup} seconds\nConnect time: %{time_connect} seconds\nStart transfer time: %{time_starttransfer} seconds\n"  -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_yolo.json

curl  -o /dev/null -s -w "\nTime taken: %{time_total} seconds\nDNS lookup time: %{time_namelookup} seconds\nConnect time: %{time_connect} seconds\nStart transfer time: %{time_starttransfer} seconds\n"  -X POST http://localhost:8000/v2/models/yolov8_combined_ensemble_onnx/versions/1/infer  \
    -H "Content-Type: application/json"  -d @input_data_yolo.json
