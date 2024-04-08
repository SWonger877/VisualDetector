from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from functools import partial
import math
import os

def get_scale():
    window = tk.Tk()
    window.state('zoomed')
    window.title('Scale Bar Value')
    title_label = ttk.Label(master=window, text='Scale Bar Value', font='Calibri 24')
    title_label.pack()

    scaleBarValueTK = tk.IntVar()
    scaleBarValueTK.set(500)
    polyFrame = ttk.Frame(master=window)
    aFrame = ttk.Frame()
    panel = tk.Label(master=aFrame)
    nextLabel = ttk.Label(master=polyFrame, text='Enter integer value for length of scale bar:')
    scaleEntry = ttk.Entry(master=polyFrame, textvariable=scaleBarValueTK)
    button1 = ttk.Button(master=polyFrame, text='All Set!', command=window.destroy)

    polyFrame.pack()
    nextLabel.pack()
    scaleEntry.pack()
    button1.pack()
    aFrame.pack()
    panel.pack()

    window.mainloop()

    return scaleBarValueTK.get()


def thresh_callback_tester(lengthLimit, scaleBarValue):
    window = tk.Tk()
    window.state('zoomed')
    window.title('Threshold Tester')
    title_label = ttk.Label(master=window, text='Threshold Tester', font='Calibri 24')
    title_label.pack()

    thresholdValue = tk.IntVar()
    thresholdValue.set(100)
    polyFrame = ttk.Frame(master=window)
    aFrame = ttk.Frame()
    panel = tk.Label(master=aFrame)
    polyTitle = ttk.Label(master = polyFrame, text = 'Threshold Value:')
    polyEntry = ttk.Entry(master=polyFrame, textvariable= thresholdValue)
    polyButton = ttk.Button(master=polyFrame, text='Test', command= partial(thresh_callback_tester_helper, lengthLimit, window, panel, thresholdValue, scaleBarValue))
    button1 = ttk.Button(master = polyFrame, text = 'Looks good, Continue!', command = window.destroy)

    polyFrame.pack()
    polyTitle.pack()
    polyEntry.pack()
    polyButton.pack()
    button1.pack()
    aFrame.pack()
    panel.pack()

    window.mainloop()

    return thresholdValue.get()


def thresh_callback_tester_helper(lengthLimit, window, panel, thresholdValue, scaleBarValue):
    src_tester = cv.imread(cv.samples.findFile(args.input))
    threshold = thresholdValue.get()
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = [x for x in contours if cv.arcLength(x, False) > 60 and cv.contourArea(x) > 10]

    contours_poly = []
    boundRect = []
    replacement = []
    for i, c in enumerate(contours):
        contours_poly.append(cv.approxPolyDP(c, 3, True))
        temp = cv.boundingRect(contours_poly[i])
        if (temp[2] > lengthLimit or temp[3] > lengthLimit) and temp[2] < len(src_tester[:][0]) / 2:
            # Catch overlapping rectangles
            if overlapCheck(temp, boundRect):
                pass
            else:
                boundRect.append(temp)
            replacement.append(c)
    contours = replacement
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    draw_on_Image_Tester(contours, drawing, boundRect, window, panel, src_tester, scaleBarValue)


def draw_on_Image_Tester(contours, drawing, boundRect, window, panel, src_tester, scaleBarValue):
    global pixel_length

    for i in range(len(contours)):
        # Draws the contours
        cv.drawContours(drawing, contours, i, (255, 255, 255))
        # Draws the convex hull
        # cv.drawContours(drawing, hull_list, i, color)
        # Draw on original image
        cv.drawContours(src_tester, contours, i, (255, 0,  0))
    for i in range(len(boundRect)):
        if i == 0:
            horz = int(boundRect[i][2])
            pixel_length = scaleBarValue/horz
        # Draw bounding box
        cv.rectangle(src_tester, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (255, 255, 255), 2)

    resizeForFit = image_resize(src_tester, 700, 700)
    # os.path.join(os.path.join(os.getcwd(), "Processing Files"), "tester.png")
    cv.imwrite('tester.png', resizeForFit)
    img = Image.open('tester.png')
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img
    panel.pack()


def thresh_callback(thres):
    threshold = thres
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filer out small sizes
    contours = [x for x in contours if cv.arcLength(x, False) > 60 and cv.contourArea(x) > 10]

    # Bounding Box
    # Could refactor to use [None]*len(contours) and filter out small bounding boxes later
    # This would change time complexity from O(n)/n (amortized O(1) in .append()) to O(1) (Java-style memory allocation)
    contours_poly = []
    boundRect = []
    replacement = []
    for i, c in enumerate(contours):
        contours_poly.append(cv.approxPolyDP(c, 3, True))
        temp = cv.boundingRect(contours_poly[i])
        if (temp[2] > lengthLimit or temp[3] > lengthLimit) and temp[2] < len(src[:][0])/2: # last check is a brute force way to remove stupidly large boxes that enclose the entire image for some reason
            # Catch overlapping rectangles
            if overlapCheck(temp, boundRect):
                pass
            else:
                boundRect.append(temp)
            replacement.append(c)
    contours = replacement

    # Find the convex hull object for each contour
    # hull_list = []
    # for i in range(len(contours)):
    #     hull = cv.convexHull(contours[i])
    #     hull_list.append(hull)

    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    folded = dict()
    individual = dict()
    openCounter = dict()
    allPolyLen = dict()
    boxesToDelete = list()
    imageCrop(contours, drawing, boundRect, folded, individual, allPolyLen, openCounter, boxesToDelete)

    # Draw on Final Output Image
    drawOnImage(contours, drawing, boundRect, folded, individual, openCounter, allPolyLen, boxesToDelete)

    # Show in a window
    window = tk.Tk()
    window.state('zoomed')
    window.title('Final Output')

    TOTALFOLD = tk.StringVar()

    TOTALFOLD.set(f'TOTAL FOLDED: {str(sum(folded.values()))} \nTOTAL INDIVIDUAL: {sum(individual.values())}\nTOTAL OPEN: {sum(openCounter.values())}\nTOTAL MER: {sum([len(i) for i in allPolyLen.values()])}')

    title_label = ttk.Label(master=window, text='FINAL OUTPUT', font='Calibri 24')
    title_label.pack()

    # Inputs
    output_label = ttk.Label(master=window, text='Folded', textvariable= TOTALFOLD)
    output_label.pack()

    srcFinal = image_resize(src, 700, 700)
    cv.imwrite('tempSrc.png', srcFinal)
    img = Image.open('tempSrc.png')
    img = ImageTk.PhotoImage(img)
    image_label = tk.Label(window, image=img)
    image_label.pack()

    window.mainloop()

    print(f'TOTAL Folded: {sum(folded.values())}')
    print(f'TOTAL Unfolded: {sum(individual.values())}')
    print(f'TOTAL Open : {sum(openCounter.values())}')
    print(f'TOTAL -mer: {sum([len(i) for i in allPolyLen.values()])}')


def drawOnImage(contours, drawing, boundRect, folded, individual, openCounter, allPolyLen, boxesToDelete):
    for i in range(len(contours)):
        color = (255, 255, 255)
        # Draws the contours
        cv.drawContours(drawing, contours, i, color)
        # Draws the convex hull
        # cv.drawContours(drawing, hull_list, i, color)
        # Draw on original image
        cv.drawContours(src, contours, i, (255, 0,  0))
    for i in range(len(boundRect)):
        if i in boxesToDelete:
            continue
        else:
            multipleCounter = 0
            if folded[i] > 0:
                color = (255, 0, 0)
                multipleCounter += 1
            if individual[i] > 0:
                color = (0, 255, 0)
                multipleCounter += 1
            if openCounter[i] > 0:
                color = (0, 0, 255)
                multipleCounter += 1
            if len(allPolyLen[i]) > 0:
                color = (255, 255, 255)
                multipleCounter += 1
            if multipleCounter == 0:
                color = (0, 0, 0)
            if multipleCounter >= 2:
                color = (100, 0, 100)
        # Draw bounding box
        cv.rectangle(src, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)


def imageCrop(contours, drawing, boundRect, folded, individual, allpolyLen, openCounter, boxesToDelete):
    for i in range(len(boundRect)):
        y, h, x, w = int(boundRect[i][1]) - 100, int(boundRect[i][3]) + 200, int(boundRect[i][0]) - 100, int(boundRect[i][2]) + 200
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        if y + h > len(drawing[:,1]):
            h = len(drawing[:,1]) - y
        if x + w > len(drawing[1,:]):
            w = len(drawing[1,:]) - x


        src_copy = cv.imread(cv.samples.findFile(args.input))
        cv.rectangle(src_copy, (int(boundRect[i][0]), int(boundRect[i][1])),
                     (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), (255, 255, 255), 1)
        crop_img = src_copy[y:y + h, x:x + w]
        crop_img = image_resize(crop_img, height=650)


        # Tkinter Window
        window = tk.Tk()
        window.state('zoomed')
        window.title('Cropped Image')

        foldInt = tk.IntVar()
        unfoldInt = tk.IntVar()
        openInt = tk.IntVar()
        polyInt = tk.IntVar()
        merVar = tk.StringVar()
        polyLen = []
        delFlagBool = [False] # Wrap in a list to force pass-by-reference

        outFold = tk.StringVar()
        outunFold = tk.StringVar()
        outOpen = tk.StringVar()
        outLengths = tk.StringVar()
        outLengthArray = tk.StringVar()
        delFlag = tk.StringVar()
        boundingBoxSize = tk.StringVar()
        boundingBoxSize.set(f'WIDTH: {boundRect[i][2]*(pixel_length):.2f}\t\t HEIGHT: {boundRect[i][3]*(pixel_length):.2f}\t\t DIAGONAL: {math.sqrt((boundRect[i][2]*(pixel_length))**2 + (boundRect[i][3]*(pixel_length))**2):.2f}')

        # Title
        title_label = ttk.Label(master = window, text = 'Cropped Section', font = 'Calibri 24')
        title_label.pack()

        # Inputs
        foldedFrame = ttk.Frame(master = window)
        foldButton = ttk.Button(master = foldedFrame, text = 'Add one folded', command = partial(count, foldInt, outFold, 'Folded Count: '))
        unfoldButton = ttk.Button(master = foldedFrame, text = 'Add one unfolded', command = partial(count, unfoldInt, outunFold, 'Unfolded Count: '))
        openButton = ttk.Button(master = foldedFrame, text = 'Add an open', command = partial(count, openInt, outOpen, 'Open Count: '))
        polyFrame = ttk.Frame(master = window)
        polyEntry = ttk.Entry(master = polyFrame, textvariable=merVar)
        polyButton = ttk.Button(master = polyFrame, text = 'Add #-mer', command = partial(polyLister, polyLen, polyInt, merVar, outLengthArray, outLengths))
        badButton = ttk.Button(master = foldedFrame, text = 'Delete this bounding box', command = partial(deleteBox, delFlag, delFlagBool))
        resetButton =  ttk.Button(master = polyFrame, text = 'Reset', command = partial(resetBox, polyLen, polyInt, merVar, outLengthArray, outLengths, unfoldInt, outunFold, openInt, outOpen, foldInt, outFold, delFlagBool, delFlag))
        nextButton = ttk.Button(master = polyFrame, text = 'Next', command = partial(next, folded, individual, openCounter, allpolyLen, boxesToDelete, delFlagBool, foldInt, unfoldInt, openInt, polyLen, i, window))


        polyEntry.pack(side = 'left', padx = 10)
        polyButton.pack(side = 'left')
        foldButton.pack(side = 'left')
        unfoldButton.pack(side = 'left')
        openButton.pack(side = 'left')
        badButton.pack()
        foldedFrame.pack()
        polyFrame.pack()
        resetButton.pack()
        nextButton.pack(side = 'right')

        # Outputs
        output_label = ttk.Label(master = window, text = 'Folded', textvariable= outFold)
        output_label1 = ttk.Label(master=window, text='Unfolded', textvariable=outunFold)
        output_label2 = ttk.Label(master=window, text='Open', textvariable=outOpen)
        output_label3 = ttk.Label(master=window, text='Polymer Count', textvariable=outLengths)
        output_label4 = ttk.Label(master = window, text = 'Polymer List', textvariable=outLengthArray)
        output_label5 = ttk.Label(master = window, text = 'Deletion', textvariable = delFlag)
        output_label6 = ttk.Label(master = window, text = 'Deletion',  textvariable = boundingBoxSize)

        output_label.pack()
        output_label1.pack()
        output_label2.pack()
        output_label3.pack()
        output_label4.pack()
        output_label5.pack()
        output_label6.pack()


        # Image
        cv.imwrite('Cropped.png', crop_img)
        img = Image.open('Cropped.png')
        img = ImageTk.PhotoImage(img)
        image_label = tk.Label(window, image = img)
        image_label.pack()
        # Run Window
        window.mainloop()


def next(folded, individual, openCounter, allpolyLen, boxesToDelete, delFlagBool, foldInt, unfoldInt, openInt, polyLen, i, window):
    # Add to running total
    if not delFlagBool[0]:  # Wrapped in a list so must index for it
        folded[i] = foldInt.get()
        individual[i] = unfoldInt.get()
        openCounter[i] = openInt.get()
        allpolyLen[i] = polyLen
    else:
        boxesToDelete.append(i)
    print(f'Updated Folded: {sum(folded.values())}')
    print(f'Updated Unfolded: {sum(individual.values())}')
    print(f'Updated Open : {sum(openCounter.values())}')
    print(f'Updated -mer: {sum([len(i) for i in allpolyLen.values()])}')
    print(f'Boxes to Delete: {boxesToDelete}')

    window.destroy()


def deleteBox(delFlag, delFlagBool):
    delFlagBool[0] = True
    delFlag.set('BOX WILL BE DELETED')


def count(iterable, output, name):
    iterable.set(iterable.get() + 1)
    output.set(f'{name}  {iterable.get()}')


def polyLister(allList, polyInt, merVar, outLenArray, outLengths):
    allList.append(merVar.get())
    try:
        polyInt.set(polyInt.get() + 1)
    except:
        pass
    outLengths.set(f'Number of -mers: {polyInt.get()}')
    outLenArray.set(f'{outLenArray.get()} {merVar.get()}-mer    ')


def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def overlapCheck(newRect, allRects):
    """Returns True if newRect overlaps, False if not. Uses the Separating Axis Theorem (where at least one side of one rectangle must act as the separating axis)"""
    # Corners of new rectangle
    x1, y1, x2, y2, = newRect[0], newRect[1], newRect[0] + newRect[2], newRect[1] + newRect[3]
    for i in allRects:
        # Corner of current pair-wise checked rectangle
        xR1, yR1, xR2, yR2 = i[0], i[1], i[0] + i[2], i[1] + i[3]
        if not (x1 > xR2
                    or x2 < xR1
                    or y1 > yR2
                    or y2 < yR1):
            allRects.remove(i) # Removes other overlapping rect
            # Adds a new rectangle that is the largest of both current rects
            # Also...mixing types in a list...OH NOOOO...whatever LOL
            fixedRect = [min(x1, xR1), min(y1, yR1), max(x2, xR2) - min(x1, xR1), max(y2, yR2) - min(y1, yR1)]
            if not overlapCheck(fixedRect, allRects):  # Recursive check to see if new rectangle overlaps with any old ones
                allRects.append(fixedRect)
            return True

    else:
        return False


def resetBox(polyLen, polyInt, merVar, outLengthArray, outLengths, unfoldInt, outunFold, openInt, outOpen, foldInt, outFold, delFlagBool, delFlag):
    # In hindsight, could have put all into an array then just for loop through w/ some if statements
    # to make it more general and efficient as well, maybe using a list wrapper around variables to force pass by reference
    # too. Oh well.
    for i in polyLen:
        polyLen.remove(i)
    polyInt.set(0)
    merVar.set('')
    outLengthArray.set('')
    outLengths.set('')
    unfoldInt.set(0)
    outunFold.set('')
    openInt.set(0)
    outOpen.set('')
    foldInt.set(0)
    outFold.set('')
    delFlagBool[0] = False
    delFlag.set('')


# Find Image to Use
cwd = os.getcwd()
path = os.path.join(cwd, "Inputs")
fileName = os.listdir(path)
filePath = os.path.join(path, fileName[0])
outputPath = os.path.join(os.path.join(cwd, "Outputs"), "temp.txt")

# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help=filePath, default=filePath)
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to grayscale
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# This line is a low pass filter that removes noise but loses edges
src_gray = cv.blur(src_gray, (3,3))
# src_gray = cv.bilateralFilter(src,9,75,75)
# Create Window
source_window = 'Source'
cv.namedWindow(source_window, cv.WINDOW_NORMAL)
cv.imshow(source_window, src)
cv.waitKey()
lengthLimit = 35 # Filter length
scaleBarValue = get_scale()
thresh = thresh_callback_tester(lengthLimit, scaleBarValue)
print(f'Threshold: {thresh}')
# cv.createTrackbar('Threshold for Dark/Light:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()