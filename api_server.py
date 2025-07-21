from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import date, timedelta
import joblib

# Khởi tạo FastAPI app
app = FastAPI(
    title="API Dự báo Tiêu thụ Nước (Full Features)",
    description="API cung cấp dự báo lượng nước tiêu thụ hàng ngày dựa trên mô hình Prophet và các yếu tố thời tiết, ngày."
)

# Tải mô hình đã huấn luyện
try:
    model_filename = 'prophet_water_forecasting_full_model.pkl'
    prophet_model = joblib.load(model_filename)
    print(f"Đã tải mô hình {model_filename} thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file mô hình '{model_filename}'. Vui lòng đảm bảo bạn đã huấn luyện và lưu mô hình.")
    prophet_model = None


# Định nghĩa Pydantic model cho dữ liệu đầu vào của API
# Mỗi ngày cần dự đoán, chúng ta cần cung cấp các regressors
class DailyForecastInput(BaseModel):
    date: date
    avg_temperature: float
    avg_humidity: float
    rainfall_mm: float
    is_holiday: int  # 0 hoặc 1
    is_weekend: int  # 0 hoặc 1 (nếu bạn thêm DayType thành Is_Weekend)


class PredictionRequest(BaseModel):
    forecast_days: list[DailyForecastInput]  # Danh sách các ngày và dữ liệu tương ứng


# Định nghĩa endpoint API để dự đoán
@app.post("/predict_water_consumption_full/")
async def predict_water_consumption_full(request: PredictionRequest):
    if prophet_model is None:
        raise HTTPException(status_code=500, detail="Mô hình dự báo chưa được tải. Vui lòng kiểm tra file mô hình.")

    if not request.forecast_days:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất một ngày để dự đoán.")

    # Chuyển đổi dữ liệu đầu vào từ Pydantic models sang Pandas DataFrame
    data_for_prophet = []
    for day_input in request.forecast_days:
        data_for_prophet.append({
            'ds': day_input.date,
            'AvgTemperature': day_input.avg_temperature,
            'AvgHumidity': day_input.avg_humidity,
            'Rainfall_mm': day_input.rainfall_mm,
            'Is_Holiday': day_input.is_holiday,
            'Is_Weekend': day_input.is_weekend
        })

    future_df = pd.DataFrame(data_for_prophet)

    # Đảm bảo cột 'ds' là datetime
    future_df['ds'] = pd.to_datetime(future_df['ds'])

    # Thực hiện dự đoán
    try:
        forecast = prophet_model.predict(future_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi thực hiện dự đoán: {str(e)}")

    results = []
    for index, row in forecast.iterrows():
        results.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_consumption": round(max(0, row['yhat']), 2),
            "lower_bound": round(max(0, row['yhat_lower']), 2),
            "upper_bound": round(max(0, row['yhat_upper']), 2)
        })

    return {"predictions": results}

# Để chạy API này, mở terminal trong thư mục chứa file api_server.py và chạy lệnh:
# uvicorn api_server:app --reload --port 8000
# Sau đó, bạn có thể truy cập tài liệu Swagger UI tại: http://127.0.0.1:8000/docs