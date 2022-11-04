import cv2
import numpy as np
from imutils import contours
import pytesseract
import os

# load image
image = cv2.imread('ex2.jpg')

#process iamge
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
outer_box = cv2.bitwise_not(gray)
outer_box = cv2.adaptiveThreshold(outer_box, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

cv2.imshow('image', outer_box)
cv2.waitKey(0)
cv2.destroyAllWindows()

#extract horizontal and vertical lines
horizontal = np.copy(outer_box)
vertical = np.copy(outer_box)

#horizontal
cols = horizontal.shape[1]
horizontal_size = cols // 10

horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

horizontal = cv2.erode(horizontal, horizontalStructure)
horizontal = cv2.dilate(horizontal, horizontalStructure)

cv2.imshow('horizontal lines', horizontal)
cv2.waitKey(0)
cv2.destroyAllWindows()

#vertical
rows = vertical.shape[0]
vertical_size = rows // 10

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

vertical = cv2.erode(vertical, verticalStructure)
vertical = cv2.dilate(vertical, verticalStructure)

cv2.imshow('vertical lines', vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()

#concatenate horizontal and vertical
concatenate = cv2.bitwise_or(vertical, horizontal)

cv2.imshow('concatenate', concatenate)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_img = cv2.bitwise_not(concatenate)

edges = cv2.adaptiveThreshold(new_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

cv2.imshow('concatenated', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)

cv2.imshow('dilated', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

smooth = cv2.blur(edges, (2, 2))

cv2.imshow('smooth edges', smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()

#find countours of detected edges
cnts, hierarchy = cv2.findContours(image=smooth, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
cp = image.copy()
cv2.drawContours(image=cp, contours=cnts, contourIdx=-1, color=(0,0,255), thickness=3, lineType=cv2.LINE_AA)

cv2.imshow('detedted contours', cp)
cv2.waitKey(0)
cv2.destroyAllWindows()

#sort extracted contours
cnts,_ = contours.sort_contours(cnts, method='left-to-right')
cnts,_ = contours.sort_contours(cnts, method='top-to-bottom')

#plot contours of each cell forming bounding box
cp = image.copy()
count = 0
for c in cnts:
    if(cv2.contourArea(c)>1000 and cv2.contourArea(c)<6000):
        count += 1
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(cp,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imshow('image', cp)
        cv2.waitKey(50)
cv2.destroyAllWindows()

def convert_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def get_num(img):
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    return pytesseract.image_to_string(img,config=r'--psm 6 --oem 3 outputbase digits -c tessedit_char_whitelist=0123456789')

cp = image.copy()
count = 0
num= []
for c in cnts:
    if(cv2.contourArea(c)>1000 and cv2.contourArea(c)<5000):
        count = count + 1
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        arr = np.array(cp[y:y+h,x:x+w])
        num.append(get_num(arr))

len(num)

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
matrix = list(divide_chunks(num, 9))
print(matrix)

matrix[0][3]

import copy

mat = copy.deepcopy(matrix)

#change '' to '0'
for n in mat:
    for j,i in enumerate(n):
        if i == '':
            n[j]=i.replace('','0')
        else:
            n[j]=i.replace(i,i[0])

mat

#change '-' to '0'
for n in mat:
    for j,i in enumerate(n):
        if i == '-':
            n[j]=i.replace('-','0')
        else:
            n[j]=i.replace(i,i[0])

mat

#change '.' to '0'
for n in mat:
    for j,i in enumerate(n):
        if i == '.':
            n[j]=i.replace('.','0')
        else:
            n[j]=i.replace(i,i[0])

mat

#change str('') into int. Ex: '5' -> 5
for row in mat:
    for i,j in enumerate(row):
        row[i] = int(j)

mat

grid = copy.deepcopy(mat)


# #solving
# def possible(row, column, number):
#     global grid
#     
#     #check if the number appearing in the given row
#     for i in range(0,9):
#         if grid[row][i] == number:
#             return False
#     
#     #check if the number appearing in the given column
#     for i in range(0,9):
#         if grid[i][column] == number:
#             return False
#     
#     #check if the number appearing in the given square
#     x0 = (column // 3) * 3
#     y0 = (row // 3) * 3
#     for i in range(0,3):
#         for j in range(0,3):
#             if grid[y0+i][x0+j] == number:
#                 return False
# 
#     return True
# 
# def solve():
#     global grid
#     for row in range(0,9):
#         for column in range(0,9):
#             if grid[row][column] == 0:
#                 for number in range(1,10):
#                     if possible(row, column, number):
#                         grid[row][column] = number
#                         solve()
#                         grid[row][column] = 0
# 
#                 return
#       
#     print(np.matrix(grid))
# solve()

M = 9
def solve(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
        
    for x in range(9):
        if grid[x][col] == num:
            return False
    
    x0 = row - row % 3
    y0 = col - col % 3
    
    for i in range(3):
        for j in range(3):
            if grid[i+x0][j+y0] == num:
                return False
    return True

def Sudoku(grid, row, col):
    if(row == M - 1 and col == M):
        return True
    if col == M:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return Sudoku(grid, row, col + 1)
    for num in range(1, M + 1, 1):
        if solve(grid, row, col, num):
            grid[row][col] = num
            if Sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

if(Sudoku(grid,0,0)):
    print(grid)
else:
    print("Sorry! :(")

bound = []
for c in cnts:
    if(cv2.contourArea(c)>1000 and cv2.contourArea(c)<5000):
        count += 1
        rect = cv2.boundingRect(c)
        bound.append(rect)

len(bound)

grid

mat

bound = list(divide_chunks(bound, 9))

cp = image.copy()
for i,m in enumerate(mat):
    for j,n in enumerate(m):
        if(mat[i][j] != grid[i][j]):
            x,y,w,h = bound[i][j]
            cv2.putText(cp,str(grid[i][j]),(x+w-35,y+h-15),2,1.3,(0,0,255))

cv2.imshow('im',cp)
cv2.waitKey(0)
cv2.destroyAllWindows()