```markdown
# API Documentation

## Get Netflix Data

### Endpoint
```

GET /api/netflix?page=1

```

### Parameters

- `page` (optional): The page number of the Netflix data. If not provided, a default message will be returned.

### Request

```

Example: http://your-api-url/api/netflix?page=1

## Get Recommendations

### Endpoint

```
GET /api/<id>/recommendations
```

### Parameters

-   `id`: The show_id of the Netflix show.

### Request

Example: GET http://your-api-url/api/70304990/recommendations

```
GET /api/netflix/<id>
```

### Parameters

-   `id`: The show_id of the Netflix show.

### Request

Example: GET http://your-api-url/api/70304990/recommendations

/////////////////////

# Hướng dẫn cài đặt thư viện

Để chạy ứng dụng Flask và sử dụng API được cung cấp, bạn cần cài đặt các thư viện cần thiết. Bạn có thể thực hiện điều này bằng cách sử dụng các bước sau:

## 1. Flask

Cài đặt Flask bằng pip:

```bash
pip install flask
```

## 2. Flask-CORS

Cài đặt Flask-CORS để xử lý Chia sẻ Tài nguyên Gốc Giữa các Trang (CORS):

```bash
pip install flask-cors
```

## 3. NetworkX

Cài đặt NetworkX để sử dụng các cấu trúc dữ liệu dựa trên đồ thị:

```bash
pip install networkx
```

## 4. Matplotlib

Cài đặt Matplotlib để thực hiện việc trực quan hóa dữ liệu:

```bash
pip install matplotlib
```

## 5. Pandas

Cài đặt Pandas để thao tác và phân tích dữ liệu:

```bash
pip install pandas
```

## 6. NumPy

Cài đặt NumPy để thực hiện các phép toán số:

```bash
pip install numpy
```

## 7. scikit-learn

Cài đặt scikit-learn để sử dụng các thuật toán và công cụ máy học:

```bash
pip install scikit-learn
```

Sau khi cài đặt các thư viện này, bạn sẽ sẵn sàng để chạy ứng dụng Flask của mình. Đảm bảo rằng bạn đã cài đặt Python trên hệ thống của mình.

## Chạy Ứng Dụng Flask

Chuyển đến thư mục chứa ứng dụng Flask của bạn và chạy lệnh sau:

```bash
python ten_ung_dung_cua_ban.py
```

Thay `ten_ung_dung_cua_ban.py` bằng tên tệp ứng dụng Flask của bạn. Điều này sẽ khởi động máy chủ phát triển Flask và bạn có thể truy cập API của mình tại các điểm cuối đã được xác định.
