# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** *11/09/2026*

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không có

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 
  ```json
  {
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
  }
  ```

- Record này mô tả toàn ảnh như thế nào?
  Gán nhãn phân loại và gán confidence score nhưng không có số lượng, bounding box, tọa độ.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Định nghĩa bởi taxonomy ImageNet-1K, bị cố định thêm bởi kích thước đầu ra.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - *Class ID:* Truy xuất, tính toán trong tensor mà không phụ thuộc vào encoding hoặc ngôn ngữ.
  - *Class Name:* Nhãn annotator đọc được.
  - *Taxonomy Name:* Tránh nhầm lẫn khi so sánh / hợp nhất.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Nếu ảnh có nhiều chủ thể cần xác định tiêu chí chọn chủ thể chính là gì, quy tắc dán nhãn và cách xử lý trường hợp không rõ ràng.

- Vì sao model score không phải ground truth?
  Model score tạo ra bởi checkpoint tại thời điểm chạy, chưa được verify bởi con người dựa trên project guideline & taxonomy.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  ```json
  {
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  }
  ```

- Diễn giải vị trí box bằng lời: 
  Vị trí tọa độ x, y tương đối của box cho thấy đối tượng chiếm nửa bên phải gần hết chiều dọc bức ảnh.

- So sánh số prediction ở hai threshold:
  Prediction threshold là 0.35 có 11 predictions, nếu tăng lên 0.5 thì số lượng sẽ giảm xuống và ngược lại.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  Nếu threshold quá cao nên sẽ nhiều obj để review, quá thấp thì sẽ quá ít để review và phải vẽ thêm bbox.

- Đề xuất một quy tắc box chặt: Góc bo chặt vật thể cần review.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  Có bao nhiêu phần trăm có thể bỏ qua, vẽ box chỉ phần nhìn thấy hay cả phần bị che mờ? Nếu không rõ ràng cần quy trình escalation để đánh giá.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  ```json
  {
    "instance_id": "kitchen-001",
    "class_name": "person",
    "score": 0.899318,
    "polygon_point_count": 348,
    "polygon_xy": [
      [446.0, 70.0],
      [445.0, 71.0],
      "..."
    ]
  }
  ```

- Polygon bổ sung chi tiết gì so với box? 
  Polygon ôm sát đường viền của vật thể ở cấp độ pixel, giúp loại bỏ phần background (hậu cảnh) và cung cấp thông tin chính xác về hình dáng, diện tích thực tế của vật thể.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  `instance_id` dùng để định danh riêng biệt cho từng cá thể vật thể (instance) cụ thể trong một bức ảnh (ví dụ: người số 1, người số 2). Nó KHÔNG phải là ID định danh toàn cục để theo dõi một vật thể cụ thể xuyên suốt nhiều bức ảnh/video khác nhau (tracking ID) hay định danh của nhóm lớp (class_id).

- Đề xuất một quy tắc biên mask:
  Các điểm của polygon phải bám sát mép ngoài cùng của vật thể, không lấn ra ngoài background quá 1-2 pixels và không được lẹm vào bên trong vật thể làm mất chi tiết.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Cần quy định rõ là có nên nội suy (đoán và vẽ tiếp) hình dáng của phần bị che khuất hay chỉ vẽ phần hiển thị rõ. Khi vật thể bị mờ nhòe (motion blur), cần xác định biên mask nên nằm ở phần viền trong hay viền ngoài của dải mờ.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn (class label) cấp độ ảnh (Text/ID) | Ảnh có nhiều đối tượng nhưng chỉ được chọn 1 nhãn, hoặc đối tượng khó phân loại | Chọn nhãn đại diện chính xác nhất cho đối tượng chủ đạo theo guideline | Đánh giá xem nhãn được chọn có phản ánh đúng đối tượng chính của ảnh không |
| Phát hiện vật thể | Bounding box (x, y, w, h) & class label | Box bao quá rộng, bị lẹm hoặc phân vân khi đối tượng bị che lấp một phần | Vẽ khung chữ nhật nhỏ nhất bao bọc toàn bộ phần nhìn thấy của đối tượng | Đảm bảo box bám sát biên, không dư viền, không cắt lẹm vào đối tượng và nhãn đúng |
| Instance segmentation | Đa giác (Polygon/Mask) các điểm (x, y) & class label | Đường viền vẽ không mượt, mask bị tràn ra background hoặc lẹm vùng che khuất | Chấm các điểm dọc theo đúng đường viền của đối tượng, tách biệt các vật thể lân cận | Độ chính xác của đường viền (không răng cưa, bám sát mép), không lấn sang hậu cảnh |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Không tải xuống, chia sẻ cục bộ, phát tán hoặc lưu trữ dữ liệu thô (hình ảnh, annotations) của dự án ra bên ngoài nền tảng làm việc bảo mật (workspace) dưới mọi hình thức.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Project Manager / Tech Lead

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
