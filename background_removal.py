import numpy as np
import cv2

# Parameters
blur = 21
canny_low = 15
canny_high = 150
min_area = 0.0005
max_area = 0.95
dilate_iter = 10
erode_iter = 10
mask_color = (0.0,0.0,0.0)

def convert_image_to_grayscale(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    return edges

def get_contour_info(edges):
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]
    return contour_info

def get_image_area(image):
    image_area = image.shape[0] * image.shape[1]  
    return image_area

def calculate_image_area(max_area, min_area, image_area):
    max_area = max_area * image_area
    min_area = min_area * image_area
    return max_area, min_area

def masking(image, edges, contour_info, mask_dilate_iter, mask_erode_iter):
    mask = np.zeros(edges.shape, dtype = np.uint8)
    for contour in contour_info:            
        if contour[1] > min_area and contour[1] < max_area:
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
    mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
    mask = cv2.erode(mask, None, iterations=mask_erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask_stack = mask.astype('float32') / 255.0           
    image = image.astype('float32') / 255.0
    masked = (mask_stack * image) + ((1-mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')        
    cv2.imshow("Foreground", masked)
    
def bg_removal(image,max_area, min_area, mask_dilate_iter, mask_erode_iter):
    image_area = get_image_area(image)
    edges = convert_image_to_grayscale(image)
    get_contour = get_contour_info(edges)
    max_area, min_area = calculate_image_area(image, max_area, min_area, image_area)
    masking(image, edges, get_contour, mask_dilate_iter, mask_erode_iter)