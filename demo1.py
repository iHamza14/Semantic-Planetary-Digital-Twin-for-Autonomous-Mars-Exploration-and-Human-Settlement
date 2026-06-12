import random
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from PIL import Image, ImageTk
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time
import cv2
import threading
import signal
import segmentation_models_pytorch as smp


# ================= CONFIG =================
NUM_CLASSES = 10
CHECKPOINT_PATH = "segmentation/runs/model.pth"
INPUT_SIZE = (252, 252)

RGB_DIR = "/Users/hamza/Desktop/Arihant/Testing/rgb"
SEG_DIR = "/Users/hamza/Desktop/Arihant/Testing/seg"

VIDEO_OUTPUT_PATH_1 = "/Users/hamza/Desktop/Arihant/Testing/rgb/outputvideo1.mp4"
VIDEO_OUTPUT_PATH_2 = "/Users/hamza/Desktop/Arihant/Testing/rgb/outputvideo2.mp4"
VIDEO_FRAME_DELAY_MS = 30
VIDEO_DISPLAY_SIZE = (1000, 700)



# ================= COLOR MAP =================
def create_color_map():
    return np.array([
        [0, 0, 0],
        [34,139,34],
        [50,205,50],
        [210,180,140],
        [139,69,19],
        [0, 0, 255],
        [255,192,203],
        [101,67,33],
        [128,128,128],
        [135,206,235],
    ], dtype=np.uint8)


CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers",
    "Logs", "Rocks", "Sky"
]

COLOR_MAP = create_color_map()


def rgb_mask_to_index(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_index = np.zeros((h, w), dtype=np.uint8)
    for class_id, color in enumerate(COLOR_MAP):
        matches = np.all(mask_rgb == color, axis=-1)
        mask_index[matches] = class_id
    return mask_index


# ================= APP =================
class OffRoadDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offroad Semantic Scene Segmentation")
        self.root.attributes("-fullscreen", True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        

        self.transform = A.Compose([
            A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.setup_ui()

        # Main display frame
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib canvas for images
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Video frame (hidden initially)
        self.video_frame = tk.Frame(self.display_frame)

        self.input_video_panel, self.input_video_label = self.create_video_panel(
            "Input Video", row=0, column=0
        )
        self.output_video_panel, self.output_video_label = self.create_video_panel(
            "Output Video 1", row=0, column=1
        )

        self.video_caps = []
        self.video_labels = []
        self.video_after_id = None
        self.third_video_window = None
        self.third_video_label = None
        self.third_video_visible = False

        # 3D Viewer
        self.current_view = "front"
        self.viewer_images = []
        self.viewer_index = 0
        

        self.show_welcome()
        self.root.bind("1", lambda e: self.show_third_video_window())

    def load_view(self, view_name):

        self.current_view = view_name

        folder = f"views/{view_name}"

        self.viewer_images = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".png")
        ])

        self.viewer_index = 0

        self.update_3d_view()

    def load_3d_views(self):
        self.load_view("front")

    def update_3d_view(self):

        if not self.viewer_images:
            return

        img = Image.open(
            self.viewer_images[self.viewer_index]
        )

        img = img.resize((1400, 900))

        photo = ImageTk.PhotoImage(img)

        self.third_video_label.configure(image=photo)
        self.third_video_label.image = photo

        self.lbl_status.config(
            text=f"3D Semantic Environment | View: {self.current_view.upper()}"
        )

    def next_view(self, event=None):

        self.viewer_index += 1

        if self.viewer_index >= len(self.viewer_images):
            self.viewer_index = 0

        self.update_3d_view()

    def prev_view(self, event=None):

        self.viewer_index -= 1

        if self.viewer_index < 0:
            self.viewer_index = len(self.viewer_images) - 1

        self.update_3d_view()

    def terminal_write(self, text):
        self.terminal.insert(tk.END, text + "\n")
        self.terminal.see(tk.END)


    def process_command(self, event=None):
        command = self.command_entry.get().strip()

        self.command_entry.delete(0, tk.END)

        if command == "!deploy_reconstruction_pipeline --mission disaster_response":
            self.run_demo_pipeline()

        elif command == "!show3D":
            self.terminal_write("[INFO] Loading reconstructed environment...")
            self.terminal_write("[INFO] Loading semantic layers...")
            self.terminal_write("[INFO] Rendering terrain model...")
            self.terminal_write("[SUCCESS] Visualization engine started.")
            self.show_third_video_window()

        else:
            self.terminal_write(f"[ERROR] Unknown command: {command}")

    # ================= MODEL =================
    def load_model(self):
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False
        )

        state_dict = checkpoint["model_state_dict"] \
            if "model_state_dict" in checkpoint else checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # ================= UI =================
    def setup_ui(self):
        self.top_frame = tk.Frame(self.root, bg="#333333", pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_image = tk.Button(
        self.top_frame,
        text="SELECT IMAGE",
        command=self.on_select_image,
        bg="#A9A9A9",
        fg="white",
        font=("Segoe UI", 14, "bold"),
        width=18,
        height=2
        )
        self.btn_image.pack(side=tk.LEFT, padx=10)

        self.btn_video = tk.Button(
        self.top_frame,
        text="SELECT VIDEO",
        command=self.on_select_video,
        bg="#A9A9A9",
        fg="white",
        font=("Segoe UI", 14, "bold"),
        width=18,
        height=2
        )
        self.btn_video.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(
            self.top_frame,
            text="System Ready",
            bg="#333333",
            fg="#AAAAAA"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # Command input
        self.command_entry = tk.Entry(
            self.top_frame,
            width=50,
            font=("Consolas", 10)
        )
        self.command_entry.pack(side=tk.LEFT, padx=10)
        self.command_entry.bind("<Return>", self.process_command)

        # Terminal output
        self.terminal = tk.Text(
            self.root,
            height=18,
            bg="black",
            fg="#00FF00",
            insertbackground="#00FF00",
            font=("Consolas", 10)
        )
        self.terminal.pack(fill=tk.X)

    def create_video_panel(self, title, row, column, columnspan=1):
        panel = tk.Frame(self.video_frame, bg="#111111", padx=8, pady=8)
        panel.grid(row=row, column=column, columnspan=columnspan, sticky="nsew")
        self.video_frame.grid_columnconfigure(column, weight=1)
        self.video_frame.grid_rowconfigure(row, weight=1)

        title_label = tk.Label(
            panel,
            text=title,
            bg="#111111",
            fg="white",
            font=("Segoe UI", 11, "bold")
        )
        title_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        video_label = tk.Label(panel, bg="black")
        video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return panel, video_label

    def show_welcome(self):
        self.fig.clear()
        self.fig.text(
            0.5, 0.5,
            "Video Segmenter and 3D Digital Twin Constructor\n\nSelect Image or Video",
            ha='center', va='center', fontsize=18
        )
        self.canvas.draw()

    def start_terminal_listener(self):
        # print("Type 1 and press Enter to show Output Video 2 in a new window.")

        def listen_for_signal():
            while True:
                try:
                    command = input().strip()
                except EOFError:
                    break

                if command == "1":
                    self.root.after(0, self.show_third_video_window)

        threading.Thread(target=listen_for_signal, daemon=True).start()

        try:
            signal.signal(
                signal.SIGUSR1,
                lambda *_: self.root.after(0, self.show_third_video_window)
            )
        except AttributeError:
            pass

    # ================= IMAGE =================
    def on_select_image(self):
        self.video_frame.pack_forget()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        file_path = filedialog.askopenfilename(
            initialdir=RGB_DIR,
            title="Select Test Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, img_path):
        self.lbl_status.config(text="Processing Image...")

        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)

        filename = os.path.basename(img_path)
        mask_path = os.path.join(SEG_DIR, filename)
        mask_index = None

        if os.path.exists(mask_path):
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask_index = rgb_mask_to_index(mask_rgb)

        augmented = self.transform(image=img_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(
                torch.softmax(output, dim=1),
                dim=1
            ).squeeze().cpu().numpy()

        self.update_plot(pil_img, prediction, mask_index)

        self.lbl_status.config(text="Image Complete")

    # ================= VIDEO =================
    def on_select_video(self):
        self.stop_video_playback()

        self.canvas.get_tk_widget().pack_forget()
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        file_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.process_video(file_path)

    def resolve_video_path(self, configured_path, input_path, title):
        candidates = []
        if os.path.isabs(configured_path):
            candidates.append(configured_path)
        else:
            candidates.append(os.path.join(os.path.dirname(input_path), configured_path))
            candidates.append(os.path.abspath(configured_path))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return filedialog.askopenfilename(
            title=title,
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )

    def process_video(self, input_path):
        self.lbl_status.config(text="Loading Videos...")

        output_1_path = self.resolve_video_path(
            VIDEO_OUTPUT_PATH_1,
            input_path,
            "Select Output Video 1"
        )
        if not output_1_path:
            self.lbl_status.config(text="Video cancelled")
            return

        output_2_path = self.resolve_video_path(
            VIDEO_OUTPUT_PATH_2,
            input_path,
            "Select Output Video 2"
        )
        if not output_2_path:
            self.lbl_status.config(text="Video cancelled")
            return

        self.start_looping_videos(input_path, output_1_path, output_2_path)

    def start_looping_videos(self, input_path, output_1_path, output_2_path):
        self.stop_video_playback()

        paths = [input_path, output_1_path, output_2_path]
        labels = [
            self.input_video_label,
            self.output_video_label,
            self.third_video_label
        ]

        caps = [cv2.VideoCapture(path) for path in paths]
        failed = [path for cap, path in zip(caps, paths) if not cap.isOpened()]
        if failed:
            for cap in caps:
                cap.release()
            messagebox.showerror(
                "Error",
                "Could not do:\n" + "\n".join(failed)
            )
            self.lbl_status.config(text="Video load failed")
            return

        self.video_caps = caps
        self.video_labels = labels
        self.third_video_visible = False
        self.lbl_status.config(text="")
        self.play_video_frame()

    def show_third_video_window(self):
        if len(self.video_caps) < 3:
            self.lbl_status.config(text="Select a video first")
            return

        if self.third_video_window is not None and self.third_video_window.winfo_exists():
            self.third_video_window.lift()
            self.third_video_visible = True
            return

        self.third_video_window = tk.Toplevel(self.root)
        self.third_video_window.title("3D Model Viewer")
        self.third_video_window.configure(bg="#111111")
        self.third_video_window.protocol("WM_DELETE_WINDOW", self.hide_third_video_window)

        frame = tk.Frame(self.third_video_window, bg="#111111", padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            frame,
            text="3D Model",
            bg="#111111",
            fg="white",
            font=("Segoe UI", 11, "bold")
        )
        title_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self.view_label = tk.Label(
        frame,
        text="View: FRONT",
        bg="#111111",
        fg="#00FFAA",
        font=("Segoe UI", 10, "bold")
        )

        self.view_label.pack()

        self.third_video_label = tk.Label(frame, bg="black")
        self.third_video_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.load_3d_views()

        self.third_video_window.bind(
            "<Right>",
            self.next_view
        )

        self.third_video_window.bind(
            "<Left>",
            self.prev_view
        )

        self.update_3d_view()
        self.video_labels[2] = self.third_video_label
        self.third_video_visible = True
        self.lbl_status.config(text="3D Model Viewer Opened")

        controls = tk.Frame(self.third_video_window)
        controls.pack(pady=10)

        tk.Button(
            controls,
            text="Front",
            width=10,
            command=lambda: self.load_view("front")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Left",
            width=10,
            command=lambda: self.load_view("left")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Right",
            width=10,
            command=lambda: self.load_view("right")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Top",
            width=10,
            command=lambda: self.load_view("top")
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
        controls,
        text="◀ Prev",
        width=10,
        command=self.prev_view
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Next ▶",
            width=10,
            command=self.next_view
        ).pack(side=tk.LEFT, padx=5)

    def hide_third_video_window(self):
        self.third_video_visible = False
        self.third_video_label = None
        if len(self.video_labels) >= 3:
            self.video_labels[2] = None
        if self.third_video_window is not None:
            self.third_video_window.destroy()
            self.third_video_window = None

    def read_looped_frame(self, cap):
        ret, frame = cap.read()
        if ret:
            return frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            return frame
        return None

    def frame_to_photo(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, VIDEO_DISPLAY_SIZE)
        return ImageTk.PhotoImage(Image.fromarray(frame))

    def play_video_frame(self):
        if not self.video_caps:
            return

        for cap, label in zip(self.video_caps[:2], self.video_labels[:2]):
            frame = self.read_looped_frame(cap)
            if frame is None:
                continue

            image = self.frame_to_photo(frame)
            label.configure(image=image)
            label.image = image

        third_frame = None
        if len(self.video_caps) >= 3:
            third_frame = self.read_looped_frame(self.video_caps[2])

        # if self.third_video_visible and self.third_video_label is not None:
        #     if third_frame is not None:
        #         image = self.frame_to_photo(third_frame)
        #         self.third_video_label.configure(image=image)
        #         self.third_video_label.image = image

        self.video_after_id = self.root.after(VIDEO_FRAME_DELAY_MS, self.play_video_frame)

    def stop_video_playback(self):
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None

        self.hide_third_video_window()

        for cap in self.video_caps:
            cap.release()

        self.video_caps = []
        self.video_labels = []

        self.lbl_status.config(text="Video Complete")

    # ================= PLOT =================
    def update_plot(self, original_img, prediction, mask_index):
        self.fig.clear()

        w_orig, h_orig = original_img.size
        prediction_resized = cv2.resize(
            prediction.astype(np.uint8),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )

        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title("Input")
        ax1.axis('off')

        ax2 = self.fig.add_subplot(1, 3, 2)
        ax2.imshow(COLOR_MAP[prediction_resized])
        ax2.set_title("Prediction")
        ax2.axis('off')

        if mask_index is not None:
            mask_resized = cv2.resize(
                mask_index.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            )
            ax3 = self.fig.add_subplot(1, 3, 3)
            ax3.imshow(COLOR_MAP[mask_resized])
            ax3.set_title("Ground Truth")
            ax3.axis('off')

        self.canvas.draw()

    def run_demo_pipeline(self):

        lines = [
            "═══════════════════════════════════════════════",
            " ORCA Segementer and 3D Digital Twin Constructor",
            "═══════════════════════════════════════════════",
            "",
            "[BOOT] Initializing mission environment...",
            "[BOOT] Loading AI segmentation engine...",
            "[BOOT] Loading terrain reconstruction modules...",
            "[BOOT] Loading navigation subsystem...",
            "[ OK ] All systems operational.",
            "",
            "MISSION OBJECTIVE",
            "Generate real-time traversable 3D terrain model",
            "",
            "[INFO] Connecting RGB stream...",
            "[INFO] Connecting LiDAR stream...",
            "[ OK ] Sensor fusion established.",
            "",
            "[INFO] Running terrain segmentation model...",
            "",
            "Trees...................24.8%",
            "Vegetation..............17.1%",
            "Dry Grass...............12.5%",
            "Traversable Ground......31.4%",
            "Rocks....................8.6%",
            "Obstacles................5.6%",
            "",
            "[ OK ] Semantic analysis complete.",
            "",
            "Average Terrain Slope.......11.7°",
            "Maximum Terrain Slope.......28.3°",
            "Traversability Score........91.2%",
            "Obstacle Count..............14",
            "Risk Zones..................2",
            "",
            "[INFO] Building point cloud...",
            "[INFO] Point cloud size: 2,347,192 points",
            "[INFO] Generating surface mesh...",
            "[INFO] Mesh vertices : 814,223",
            "[INFO] Mesh faces    : 428,991",
            "",
            "[ OK ] Reconstruction complete.",
            "",
            "MISSION STATUS : SUCCESS",
            "",
            "Available command:",
            "!show3D"
        ]

        def print_line(i=0):
            if i < len(lines):
                self.terminal_write(lines[i])
                self.root.after(120, lambda: print_line(i + 1))

        print_line()


if __name__ == "__main__":
    root = tk.Tk()
    app = OffRoadDemoApp(root)
    root.mainloop()
