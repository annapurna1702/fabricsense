import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.clock import Clock
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super(SplashScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.image = Image(source='pic.jpg')
        layout.add_widget(self.image)
        self.add_widget(layout)

    def on_enter(self):
        Clock.schedule_once(self.switch_to_upload, 3)

    def switch_to_upload(self, dt):
        self.manager.current = 'upload'

class UploadScreen(Screen):
    def __init__(self, **kwargs):
        super(UploadScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        self.upload_button = Button(text="Upload and Process Image", on_press=self.process_image)
        layout.add_widget(self.filechooser)
        layout.add_widget(self.upload_button)
        self.add_widget(layout)

    def process_image(self, instance):
        selected = self.filechooser.selection
        if selected:
            image_path = selected[0]
            try:
                detect_lines(image_path)
                self.manager.current = 'info'
            except Exception as e:
                print(f"Error processing image: {e}")

class InfoScreen(Screen):
    def __init__(self, **kwargs):
        super(InfoScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.name_input = TextInput(hint_text='Enter tester name')
        self.date_input = TextInput(hint_text='Enter date')
        self.submit_button = Button(text='Submit', on_press=self.submit_info)
        layout.add_widget(self.name_input)
        layout.add_widget(self.date_input)
        layout.add_widget(self.submit_button)
        self.add_widget(layout)

    def submit_info(self, instance):
        self.manager.current = 'report'
        self.manager.get_screen('report').generate_report(self.name_input.text, self.date_input.text)

class ReportScreen(Screen):
    def __init__(self, **kwargs):
        super(ReportScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.report_label = Label()
        self.save_button = Button(text='Save Report', on_press=self.save_report)
        layout.add_widget(self.report_label)
        layout.add_widget(self.save_button)
        self.add_widget(layout)

    def generate_report(self, name, date):
        self.name = name
        self.date = date
        self.report_label.text = f'Tester: {name}\nDate: {date}\nVertical Threads: {self.manager.vertical_count}\nHorizontal Threads: {self.manager.horizontal_count}\nTotal Threads: {self.manager.total_count}'

    def save_report(self, instance):
        with open('report.txt', 'w') as f:
            f.write(self.report_label.text)

class ThreadDetectionApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(SplashScreen(name='splash'))
        self.sm.add_widget(UploadScreen(name='upload'))
        self.sm.add_widget(InfoScreen(name='info'))
        self.sm.add_widget(ReportScreen(name='report'))
        return self.sm

    def on_start(self):
        self.sm.current = 'splash'
        self.sm.vertical_count = 0
        self.sm.horizontal_count = 0
        self.sm.total_count = 0

def detect_lines(image_path, angle_tolerance=10, eps=5, min_samples=2):
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply GaussianBlur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Use adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Use Canny Edge Detector with adjusted thresholds
        edges = cv2.Canny(adaptive_thresh, 50, 100, apertureSize=3)

        # Detect lines using Hough Transform with a lower threshold
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                degree_angle = np.degrees(theta)
                if (degree_angle < 0.8 or degree_angle > 180 - 0.8):
                    vertical_lines.append(line)
                elif (90 - angle_tolerance < degree_angle < 90 + angle_tolerance):
                    horizontal_lines.append(line)

        def cluster_lines(lines, eps, min_samples):
            if not lines:
                return []
            points = np.array([[line[0][0] * np.cos(line[0][1]), line[0][0] * np.sin(line[0][1])] for line in lines])
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            unique_labels = set(clustering.labels_)
            return unique_labels, clustering

        vertical_labels, vertical_clustering = cluster_lines(vertical_lines, eps, min_samples)
        horizontal_labels, horizontal_clustering = cluster_lines(horizontal_lines, eps, min_samples)
        vertical_count = len(vertical_labels) - (1 if -1 in vertical_labels else 0)
        horizontal_count = len(horizontal_labels) - (1 if -1 in horizontal_labels else 0)
        total_count = vertical_count + horizontal_count

        print(f"Estimated number of vertical threads: {vertical_count}")
        print(f"Estimated number of horizontal threads: {horizontal_count}")
        print(f"Estimated total number of threads: {total_count}")

        ThreadDetectionApp.get_running_app().sm.vertical_count = vertical_count
        ThreadDetectionApp.get_running_app().sm.horizontal_count = horizontal_count
        ThreadDetectionApp.get_running_app().sm.total_count = total_count

        # Draw the lines on the image
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Lines')
        plt.show()

    except Exception as e:
        print(f"Error in detect_lines: {e}")

if __name__ == '__main__':
    ThreadDetectionApp().run()
