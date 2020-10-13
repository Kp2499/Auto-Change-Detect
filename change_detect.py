# importing UI components
import tkinter as tk
from tkinter import filedialog, Text, messagebox
import PIL
from PIL import Image, ImageTk
import os

# importing autoChange components
from skimage.metrics import structural_similarity
import imutils
import cv2
import time
import numpy as np

root = tk.Tk()  # initializing tkinter
root.title('Auto Change Detection')
root.attributes('-fullscreen', False)
root.resizable(width=False, height=False)
root.geometry('838x400')

# initialializing Default paths
videoPath1 = "/"
videoPath2 = "/"

# getting reference video path
def addRefernceVideoHandler():
    globalRefernceVideoPath = filedialog.askopenfilename(initialdir="/", title="select file",
                                                         filetypes=(("Video Files", "*.mp4"), ("mkv files", "*.mkv")))

    if globalRefernceVideoPath:
        global videoPath1
        videoPath1 = globalRefernceVideoPath
        global refernceLabel
        refernceLabel = tk.Label(frame, text="Refernce video is at : " + globalRefernceVideoPath, bg="gray",
                                 fg='#f2f3f4')
        selectVideo2Btn.pack()
    else:
        refernceLabel = tk.Label(frame, text="Select Valid Refernce Video path", bg="red", fg='#f2f3f4')
    refernceLabel.pack()



# getting change video path
def addAutoChangeVideoHandler():
    globalAutoChangeVideoPath = filedialog.askopenfilename(initialdir="/", title="select file",
                                                           filetypes=(("Video Files", "*.mp4"), ("mkv files", "*.mkv")))

    if globalAutoChangeVideoPath:
        global videoPath2
        videoPath2 = globalAutoChangeVideoPath
        global autoChangeLabel
        autoChangeLabel = tk.Label(frame, text="Video for AutoChange is at : " + globalAutoChangeVideoPath, bg="gray",
                                   fg='#f2f3f4')
        frame3.pack()
        selectVideo2Btn.pack_forget()
        selectVideo1Btn.pack_forget()
        runAutoChangeDetection100mBtn.pack(side=tk.LEFT)
        runAutoChangeDetection200mBtn.pack(side=tk.LEFT)
        runAutoChangeDetection300mBtn.pack(side=tk.LEFT)
        onBackBtn.pack(side=tk.LEFT)

    else:
        autoChangeLabel = tk.Label(frame, text="Select Valid Video ", bg="red", fg='#f2f3f4')

    autoChangeLabel.pack()


def onCallBackHandler():
    # destroying the auto change and back button
    runAutoChangeDetection100mBtn.pack_forget()
    runAutoChangeDetection200mBtn.pack_forget()
    runAutoChangeDetection300mBtn.pack_forget()
    onBackBtn.pack_forget()
    frame3.pack_forget()
    autoChangeLabel.pack_forget()
    refernceLabel.pack_forget()
    selectVideo1Btn.pack()  # reimplementing the video one button


def callingAutoChange(option):
    # Changing UI components
    # Makes main window
    root.destroy()
    global window
    window = tk.Tk()
    window.wm_title("Auto Change Detection")
    window.attributes('-fullscreen', False)
    window.resizable(width=False, height=False)
    window.config(background="#273746")

    # Graphic Window for single frame
    imageFrame = tk.Frame(window, width=600, height=500)
    imageFrame.grid(row=0, column=0, padx=0, pady=0)
    # IMAGE DIMENSIONS
    img_x = 480
    img_y = 360
    # PARAMETER SETTING
    if option == 1:
        upper_hist_limit = 9500
        limit = 25
        score_limit = 0.60
    elif option == 2:
        upper_hist_limit = 8000
        limit = 5
        score_limit = 0.35
    elif option == 3:
        upper_hist_limit = 7000
        limit = 6
        score_limit = 0.40

    start = time.time()
    PATH_TO_REF_VIDEO = videoPath1
    PATH_TO_CHANGE_VIDEO = videoPath2
    cap_1 = cv2.VideoCapture(PATH_TO_REF_VIDEO)
    cap_2 = cv2.VideoCapture(PATH_TO_CHANGE_VIDEO)
    ref_vid = os.path.basename(PATH_TO_REF_VIDEO)
    change_vid = os.path.basename(PATH_TO_CHANGE_VIDEO)
    ref_vid = os.path.splitext(ref_vid)[0]
    change_vid = os.path.splitext(change_vid)[0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if not os.path.exists('Processed Video'):
        os.makedirs('Processed Video')
    cwd = os.getcwd()
    writer = cv2.VideoWriter(os.path.join('Processed Video', change_vid) + '_change_' + '.mp4', fourcc, 15, (img_x, img_y))

    def show_frame():
        # VIDEO PROCESSING
        _, frame_1 = cap_1.read()
        __, frame_2 = cap_2.read()
        if _ == False or __ == False:
            messagebox.showinfo('PROCESS ENDED','The video has ended and the processed video is saved at:\n' + os.path.join(cwd, 'Processed Video'))
            window.destroy()
        if _ == True and __ == True:
            # Resizing video frame-by-frame
            resized_1 = cv2.resize(frame_1, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
            resized_2 = cv2.resize(frame_2, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
            # Converting to HSV
            hsv_1 = cv2.cvtColor(resized_1, cv2.COLOR_BGR2HSV)
            hsv_2 = cv2.cvtColor(resized_2, cv2.COLOR_BGR2HSV)
            histogram_1 = cv2.calcHist([hsv_1], [0], None, [256], [0, 256])
            histogram_2 = cv2.calcHist([hsv_2], [0], None, [256], [0, 256])
            i = 0
            dist = 0
            while i < len(histogram_1) and i < len(histogram_2):
                dist += (histogram_1[i] - histogram_2[i]) ** 2
                i += 1
            dist = dist ** (1 / 2)
            bfilter_1 = cv2.bilateralFilter(hsv_1, 15, 75, 75)
            bfilter_2 = cv2.bilateralFilter(hsv_2, 15, 75, 75)
            canny_1 = cv2.Canny(bfilter_1, 55, 165, 2)
            canny_2 = cv2.Canny(bfilter_2, 55, 165, 2)
            # Compute the Structural Similarity Index between 2 images, ensuring that the difference image is returned
            (score, diff) = structural_similarity(canny_1, canny_2, full=True)
            diff = (diff * 255).astype('uint8')
            # threshold the difference image, followed by finding contours to obtain the regions of the 2 input images that differ
            thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # loop over contours
            # Compute the bounding box of contour and then draw the bounding box on both input images to represent where the 2 images differ
            info = np.zeros([360, 600, 3], dtype=np.uint8)
            info[:, :] = [0, 255,  255]
            cv2.putText(info, '# INFORMATION :', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(info, "1. Change Detection video is being", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(info, "automatically recorded and stored. ", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(info, "2. For taking ScreenShot, press 's' key .", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv2.putText(info, "3. To quit the application, press 'q' key .", (20,180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


            msg = np.zeros([240, 600, 3], dtype=np.uint8)
            msg[:, :] = [255, 255, 0]
            if option == 1:
                cv2.putText(msg, '# INFO: Processing for 100m height video.', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            elif option == 2:
                cv2.putText(msg, '# INFO: Processing for 200m height video.', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            elif option == 3:
                cv2.putText(msg, '# INFO: Processing for 300m height video.', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


            if ((dist <= upper_hist_limit and dist != 0) or (score >= score_limit and score != 1)):
                cv2.putText(msg, '# INFO:', (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.putText(msg, 'CHANGE DETECTED !', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if (w >= limit) and (h >= limit) and (w <= 240) and (h <= 180):
                        cv2.rectangle(resized_2, (x, y), (x + w, y + h), (51, 255, 255), 2)
            writer.write(resized_2)
            if (dist != 0 and score != 1) and ((dist > upper_hist_limit) and (score <= score_limit)):
                cv2.putText(msg, "# WARNING: ", (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(msg, " Video/Scene is completely different,so no detection done!", (0, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            elif dist == 0 and score == 1:
                cv2.putText(msg, "# WARNING:", (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(msg, "SAME SCENES DETECTED !", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # VIDEO OPTIONS
            end = time.time()
            t = end - start
            t = str(round(t, 2))
            window.bind('s', lambda x: screenShotHandler(change_vid,t,resized_2))
            window.bind('q', lambda x: on_closing())
            cv2image = cv2.cvtColor(resized_1, cv2.COLOR_BGR2RGBA)
            cv2image1 = cv2.cvtColor(resized_2, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img1 = Image.fromarray(cv2image1)
            img2 = Image.fromarray(canny_1)
            img3 = Image.fromarray(canny_2)
            img4 = Image.fromarray(msg)
            img5 = Image.fromarray(info)
            imgtk = ImageTk.PhotoImage(image=img)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            imgtk3 = ImageTk.PhotoImage(image=img3)
            imgtk4 = ImageTk.PhotoImage(image=img4)
            imgtk5 = ImageTk.PhotoImage(image=img5)

            display1.imgtk = imgtk  # Shows frame for display 1
            display1.configure(image=imgtk)

            display2.imgtk = imgtk1  # Shows frame for display 2
            display2.configure(image=imgtk1)

            display3.imgtk = imgtk2  # Shows frame for display 3
            display3.configure(image=imgtk2)

            display4.imgtk = imgtk3  # Shows frame for display 4
            display4.configure(image=imgtk3)

            display5.imgtk = imgtk4  # Shows frame for display 5
            display5.configure(image=imgtk4)

            display6.imgtk = imgtk5  # Shows frame for display 6
            display6.configure(image=imgtk5)

            window.after(25, show_frame)

    display1 = tk.Label(imageFrame)
    display1.grid(row=0, column=0)  # Display 1

    display2 = tk.Label(imageFrame)
    display2.grid(row=0, column=1)  # Display 2

    display3 = tk.Label(imageFrame)
    display3.grid(row=1, column=0)  # Display 3

    display4 = tk.Label(imageFrame)
    display4.grid(row=1, column=1)  # Display 4

    display5 = tk.Label(imageFrame)
    display5.grid(row=0, column=2)  # Display 5

    display6 = tk.Label(imageFrame)
    display6.grid(row=1, column=2)  # Display 6

    show_frame()
    window.protocol('WM_DELETE_WINDOW', on_closing)
    window.mainloop()

def screenShotHandler(name,sec,image):
    newpath = 'output'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    cv2.imwrite(os.path.join(newpath, name + ' auto_change ' + sec + '.jpg'),image)
    messagebox.showinfo('Screenshot', 'ScreenShot Taken')

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

def onroot_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

# Creating UI
canvas = tk.Canvas(root, height=300, width=900, bg="#263d43")
canvas.pack()
global frame
frame = tk.Frame(root, bg="#273746")
frame.place(relwidth=1, relheight=1)
# Showing Heading
mainHeading = tk.Label(frame, text="SELECT VIDEOS FOR AUTO CHANGE", font=("Helvetica", 16), fg='#f2f3f4', padx=20,
                       pady=20)
mainHeading['bg'] = mainHeading.master['bg']
mainHeading.pack()
global selectVideo1Btn
selectVideo1Btn = tk.Button(root, text="Select Refernce Video", font=("Helvetica", 10), height=2, width=20,
                            fg="black", bg="#d5d8dc",
                            command=addRefernceVideoHandler)
global selectVideo2Btn
selectVideo2Btn = tk.Button(root, text="Select Second Video", font=("Helvetica", 10), height=2, width=20,
                            fg="black", bg="#d5d8dc", command=addAutoChangeVideoHandler)
selectVideo1Btn.pack()
global runAutoChangeDetection100mBtn
runAutoChangeDetection100mBtn = tk.Button(root, text="Run Auto Change Detection for 100 meter", height=2, width=40,
                                          fg='black', bg="#f6ddcc", command=lambda: callingAutoChange(1))
global runAutoChangeDetection200mBtn
runAutoChangeDetection200mBtn = tk.Button(root, text="Run Auto Change Detection for 200 meter", height=2, width=40,
                                          fg='black', bg="#f6ddcc", command=lambda: callingAutoChange(2))
global runAutoChangeDetection300mBtn
runAutoChangeDetection300mBtn = tk.Button(root, text="Run Auto Change Detection for 300 meter", height=2, width=40,
                                          fg='black', bg="#f6ddcc", command=lambda: callingAutoChange(3))
# adding back button functionality and additional frame
global frame3
frame3 = tk.Frame()
global onBackBtn
onBackBtn = tk.Button(frame3, text='Back or Reselect', height=2, width=20, fg='black', bg='#f6ddcc',
                      command=onCallBackHandler)

root.protocol('WM_DELETE_WINDOW', onroot_closing)
root.mainloop()
