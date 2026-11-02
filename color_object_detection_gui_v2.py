
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

COLOR_RANGES = {
    "Red": [
        (np.array([0, 90, 60]), np.array([10, 255, 255])),
        (np.array([170, 90, 60]), np.array([179, 255, 255]))
    ],
    
    "Green": [
        (np.array([25, 35, 25]), np.array([95, 255, 255]))
    ],
    # เริ่ม Blue ที่ Hue 96 เพื่อไม่ให้ทับซ้อนกับช่วง Green
    "Blue": [
        (np.array([96, 45, 35]), np.array([135, 255, 255]))
    ]
}

COLOR_BGR = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0)
}


MIN_AREA = 1200


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class ColorObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Object Detection - RGB Color Counter")
        self.root.geometry("1250x900")
        self.root.minsize(1000, 760)
        self.root.configure(bg="#eef2f7")

        self.camera = None
        self.camera_running = False
        self.after_id = None
        self.photo = None

        self.current_counts = {"Red": 0, "Green": 0, "Blue": 0}
        self.round_history = []

        self.setup_style()
        self.build_interface()
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Start.TButton", font=("Segoe UI", 11, "bold"),
            padding=9, background="#1677d2", foreground="white"
        )
        style.map(
            "Start.TButton",
            background=[("active", "#0f5eaa"), ("disabled", "#9aa8b7")]
        )
        style.configure(
            "Stop.TButton", font=("Segoe UI", 11, "bold"),
            padding=9, background="#d94848", foreground="white"
        )
        style.map(
            "Stop.TButton",
            background=[("active", "#b83232"), ("disabled", "#c7a0a0")]
        )
        style.configure("Normal.TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def build_interface(self):
        header = tk.Frame(self.root, bg="#172b4d", height=76)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header, text="COLOR OBJECT DETECTION",
            font=("Segoe UI", 21, "bold"),
            fg="white", bg="#172b4d"
        ).pack(anchor="w", padx=22, pady=(9, 0))

        tk.Label(
            header, text="Real-time Red / Green / Blue detection and counting",
            font=("Segoe UI", 10), fg="#d6e2f0", bg="#172b4d"
        ).pack(anchor="w", padx=24)

        controls = tk.Frame(self.root, bg="#eef2f7")
        controls.pack(fill="x", padx=18, pady=9)

        self.start_button = ttk.Button(
            controls, text="▶  เริ่มตรวจจับ",
            style="Start.TButton", command=self.start_camera
        )
        self.start_button.pack(side="left", padx=(0, 8))

        self.stop_button = ttk.Button(
            controls, text="■  หยุดและบันทึกผล",
            style="Stop.TButton", command=self.stop_and_save,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)

        self.clear_button = ttk.Button(
            controls, text="↻  ล้างผลทั้งหมด",
            style="Normal.TButton", command=self.clear_all
        )
        self.clear_button.pack(side="left", padx=5)

        self.status_label = tk.Label(
            controls, text="สถานะ: พร้อมเริ่มตรวจจับ",
            font=("Segoe UI", 10, "bold"),
            fg="#53657d", bg="#eef2f7"
        )
        self.status_label.pack(side="right", padx=5)

        # พื้นที่ภาพกล้อง + แผงนับจำนวน
        main = tk.Frame(self.root, bg="#eef2f7", height=440)
        main.pack(fill="x", padx=18, pady=(0, 9))
        main.pack_propagate(False)

        camera_card = tk.Frame(
            main, bg="white", highlightthickness=1,
            highlightbackground="#d8e0ea"
        )
        camera_card.pack(side="left", fill="both", expand=True, padx=(0, 9))

        tk.Label(
            camera_card, text="ภาพจาก Webcam - Real-time Detection",
            font=("Segoe UI", 12, "bold"), bg="white", fg="#172b4d"
        ).pack(anchor="w", padx=12, pady=(8, 4))

        self.video_label = tk.Label(
            camera_card,
            text="กดเริ่มตรวจจับเพื่อเปิด Webcam\n"
                 "นำวัตถุสีแดง สีเขียว หรือสีน้ำเงินมาวางหน้ากล้อง",
            font=("Segoe UI", 12), bg="#182334", fg="white",
            justify="center"
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        count_card = tk.Frame(
            main, bg="white", width=280,
            highlightthickness=1, highlightbackground="#d8e0ea"
        )
        count_card.pack(side="right", fill="y")
        count_card.pack_propagate(False)

        tk.Label(
            count_card, text="จำนวนวัตถุที่ตรวจพบ",
            font=("Segoe UI", 12, "bold"), bg="white", fg="#172b4d"
        ).pack(anchor="w", padx=12, pady=(12, 8))

        self.count_labels = {}
        for name, hex_color in [
            ("Red", "#dc3545"), ("Green", "#198754"), ("Blue", "#0d6efd")
        ]:
            panel = tk.Frame(count_card, bg="#f4f7fb")
            panel.pack(fill="x", padx=10, pady=7)

            tk.Label(
                panel, text=name, font=("Segoe UI", 11, "bold"),
                fg=hex_color, bg="#f4f7fb"
            ).pack(anchor="w", padx=10, pady=(7, 0))

            value = tk.Label(
                panel, text="0 วัตถุ",
                font=("Segoe UI", 21, "bold"),
                fg="#24364b", bg="#f4f7fb"
            )
            value.pack(anchor="w", padx=10, pady=(0, 7))
            self.count_labels[name] = value

        self.total_label = tk.Label(
            count_card, text="รวมทั้งหมด: 0 วัตถุ",
            font=("Segoe UI", 11, "bold"), bg="white", fg="#172b4d"
        )
        self.total_label.pack(anchor="w", padx=12, pady=(12, 5))

        self.round_label = tk.Label(
            count_card, text="รอบถัดไป: 1",
            font=("Segoe UI", 10, "bold"), bg="white", fg="#1677d2"
        )
        self.round_label.pack(anchor="w", padx=12, pady=5)

        tk.Label(
            count_card,
            text="หลักการทำงาน:\n"
                 "1. แปลงภาพเป็น HSV\n"
                 "2. แยกช่วงสีที่กำหนด\n"
                 "3. กรองจุดรบกวน\n"
                 "4. นับวัตถุจาก Contours",
            font=("Segoe UI", 9), bg="white", fg="#65758b",
            justify="left"
        ).pack(anchor="w", padx=12, pady=10)

      
        table_card = tk.Frame(
            self.root, bg="white", height=155,
            highlightthickness=1, highlightbackground="#d8e0ea"
        )
        table_card.pack(fill="x", padx=18, pady=(0, 9))
        table_card.pack_propagate(False)

        tk.Label(
            table_card, text="ประวัติผลการตรวจจับแต่ละรอบ",
            font=("Segoe UI", 11, "bold"), bg="white", fg="#172b4d"
        ).pack(anchor="w", padx=12, pady=(6, 3))

        columns = ("round", "red", "green", "blue", "total")
        self.table = ttk.Treeview(
            table_card, columns=columns, show="headings", height=3
        )
        headings = {
            "round": "รอบ", "red": "Red (ชิ้น)",
            "green": "Green (ชิ้น)", "blue": "Blue (ชิ้น)",
            "total": "รวมทั้งหมด"
        }
        widths = {
            "round": 90, "red": 170, "green": 170,
            "blue": 170, "total": 170
        }
        for col in columns:
            self.table.heading(col, text=headings[col])
            self.table.column(col, width=widths[col], anchor="center")
        self.table.pack(fill="both", expand=True, padx=9, pady=(0, 7))

        
        self.graph_card = tk.Frame(
            self.root, bg="white",
            highlightthickness=1, highlightbackground="#d8e0ea"
        )
        self.graph_card.pack(fill="both", expand=True, padx=18, pady=(0, 13))

        self.graph_placeholder = tk.Label(
            self.graph_card,
            text="กราฟจำนวนวัตถุแต่ละสีจะแสดงหลังจากหยุดและบันทึกผล",
            font=("Segoe UI", 11), bg="white", fg="#718096"
        )
        self.graph_placeholder.pack(expand=True)

    def detect_colors(self, frame):
        """คืนภาพที่วาดกรอบแล้ว และจำนวนวัตถุของแต่ละสี"""
       
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        counts = {"Red": 0, "Green": 0, "Blue": 0}
        kernel = np.ones((5, 5), np.uint8)

        for color_name, ranges in COLOR_RANGES.items():
           
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                part = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, part)

         
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                counts[color_name] += 1

                # วาดกรอบและชื่อสีพร้อมลำดับวัตถุ
                bgr = COLOR_BGR[color_name]
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 3)

                label = f"{color_name} #{counts[color_name]}"
                text_y = max(28, y - 10)
                cv2.putText(
                    frame, label, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2,
                    cv2.LINE_AA
                )

        return frame, counts

    def start_camera(self):
        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror(
                "เปิดกล้องไม่สำเร็จ",
                "ไม่สามารถเปิด Webcam ได้ กรุณาตรวจสอบการเชื่อมต่อกล้อง"
            )
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.camera_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(
            text="สถานะ: กำลังตรวจจับสีแบบ Real-time",
            fg="#198754"
        )
        self.update_camera()

    def update_camera(self):
        if not self.camera_running or self.camera is None:
            return

        ret, frame = self.camera.read()
        if not ret:
            self.stop_camera()
            messagebox.showwarning("กล้อง", "อ่านภาพจาก Webcam ไม่สำเร็จ")
            return

        processed, counts = self.detect_colors(frame)
        self.current_counts = counts
        self.update_count_labels(counts)

        # แสดงภาพโดยรักษาสัดส่วน ไม่ยืดภาพ
        display_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display_rgb)
        image.thumbnail((900, 390))
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=self.photo, text="")

        self.after_id = self.root.after(30, self.update_camera)

    def update_count_labels(self, counts):
        for color_name, count in counts.items():
            self.count_labels[color_name].config(
                text=f"{count} วัตถุ"
            )

        total = sum(counts.values())
        self.total_label.config(text=f"รวมทั้งหมด: {total} วัตถุ")

    def stop_camera(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.camera_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def stop_and_save(self):
        if not self.camera_running:
            return

        # ใช้จำนวนจากเฟรมล่าสุดที่ประมวลผลได้
        result = {
            "round": len(self.round_history) + 1,
            "Red": int(self.current_counts["Red"]),
            "Green": int(self.current_counts["Green"]),
            "Blue": int(self.current_counts["Blue"])
        }
        result["Total"] = result["Red"] + result["Green"] + result["Blue"]

        self.stop_camera()
        self.round_history.append(result)

        self.table.insert(
            "", "end",
            values=(
                f"รอบ {result['round']}",
                result["Red"], result["Green"],
                result["Blue"], result["Total"]
            )
        )

        self.update_graph()
        self.round_label.config(
            text=f"รอบถัดไป: {len(self.round_history) + 1}"
        )
        self.status_label.config(
            text=f"สถานะ: บันทึกผลรอบที่ {result['round']} แล้ว",
            fg="#1677d2"
        )
        self.video_label.config(
            image="",
            text=(
                f"Webcam ปิดแล้ว\n"
                f"Red: {result['Red']}  Green: {result['Green']}  "
                f"Blue: {result['Blue']}"
            ),
            bg="#182334", fg="white"
        )
        self.photo = None

    def update_graph(self):
        for widget in self.graph_card.winfo_children():
            widget.destroy()

        rounds = [item["round"] for item in self.round_history]
        x = np.arange(len(rounds))
        width = 0.24

        fig = Figure(figsize=(11, 3.4), dpi=95)
        ax = fig.add_subplot(111)

        red_values = [item["Red"] for item in self.round_history]
        green_values = [item["Green"] for item in self.round_history]
        blue_values = [item["Blue"] for item in self.round_history]

        bars_red = ax.bar(
            x - width, red_values, width,
            color="#dc3545", label="Red"
        )
        bars_green = ax.bar(
            x, green_values, width,
            color="#198754", label="Green"
        )
        bars_blue = ax.bar(
            x + width, blue_values, width,
            color="#0d6efd", label="Blue"
        )

        ax.bar_label(bars_red, padding=2, fontsize=8)
        ax.bar_label(bars_green, padding=2, fontsize=8)
        ax.bar_label(bars_blue, padding=2, fontsize=8)

        ax.set_title("Number of Detected Objects by Color", fontsize=11)
        ax.set_xlabel("Measurement Round")
        ax.set_ylabel("Number of Objects")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Round {r}" for r in rounds])
        ax.set_ylim(
            0, max(1, max(red_values + green_values + blue_values) * 1.25)
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_card)
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=8, pady=6
        )

        tk.Label(
            self.graph_card,
            text="กราฟแสดงจำนวนวัตถุที่ตรวจพบในแต่ละรอบ (หน่วย: ชิ้น)",
            font=("Segoe UI", 8), bg="white", fg="#53657d"
        ).pack(pady=(0, 4))

    def clear_all(self):
        if self.camera_running:
            messagebox.showwarning(
                "กล้องยังเปิดอยู่",
                "กรุณาหยุดกล้องก่อนล้างข้อมูล"
            )
            return

        if self.round_history and not messagebox.askyesno(
            "ยืนยัน", "ต้องการลบประวัติการตรวจจับทั้งหมดใช่หรือไม่?"
        ):
            return

        self.round_history.clear()
        self.current_counts = {"Red": 0, "Green": 0, "Blue": 0}
        self.update_count_labels(self.current_counts)

        for item in self.table.get_children():
            self.table.delete(item)

        self.round_label.config(text="รอบถัดไป: 1")
        self.status_label.config(
            text="สถานะ: พร้อมเริ่มตรวจจับ", fg="#53657d"
        )
        self.video_label.config(
            image="",
            text="กดเริ่มตรวจจับเพื่อเปิด Webcam\n"
                 "นำวัตถุสีแดง สีเขียว หรือสีน้ำเงินมาวางหน้ากล้อง",
            bg="#182334", fg="white"
        )
        self.photo = None

        for widget in self.graph_card.winfo_children():
            widget.destroy()

        self.graph_placeholder = tk.Label(
            self.graph_card,
            text="กราฟจำนวนวัตถุแต่ละสีจะแสดงหลังจากหยุดและบันทึกผล",
            font=("Segoe UI", 11), bg="white", fg="#718096"
        )
        self.graph_placeholder.pack(expand=True)

    def close_application(self):
        self.stop_camera()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorObjectDetector(root)
    root.mainloop()
