# LOAD LIBRARIES
import numpy as np
import tensorflow as tf
import pygame
import cv2
import os
import sys
from pygame.locals import*

#INITIALIZE PYGAME
pygame.init()

#VARIABLES USED
screen_width = 800
screen_height = 800
white = (255,255,255)
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)
exit = False
pred = False
save_image = False
write = False
bound  = 20
font = pygame.font.Font("freesansbold.ttf", 18)
x_coord = []
y_coord = []
n_image = 1
save_image = True
model = tf.keras.models.load_model("MNIST-Model(acc-97%).h5")
LABEL = {
	0: 'ZERO',
	1: "ONE",
	2: "TWO",
	3: "THREE",
	4: "FOUR",
	5: "FIVE",
	6: "SIX",
	7: "SEVEN",
	8: "EIGHT",
	9: "NINE",
}
#CREATE FOLDER FOR IMAGES TO STORE, AFTER CREATING IT COMMENT THIS CODE
# os.path.isdir("images")
# os.mkdir("images")

# CREATING CANVAS FOR DISPLAY NUMBERS
canvas = pygame.display.set_mode((screen_width,screen_height))

# BACKGROUND OF CANVAS CHANGES TO WHITE
# canvas.fill(white)

#TITLE OF CANVAS
pygame.display.set_caption("MNIST Board")

#CREATING LOOP FOR INFINIE TIMES
while not exit:

	for event in pygame.event.get():

		if event.type == pygame.QUIT:

			exit = True

		# ON HOVERING MOUSE
		if event.type == MOUSEMOTION and write:
			x,y = event.pos
			x_coord.append(x)
			y_coord.append(y)	
			pygame.draw.circle(canvas,white,(x,y),5,0)
		# ON CLICKING ON MOUSE BUTTON
		if event.type == MOUSEBUTTONDOWN:
			write = True
		# AFTER CLICKING ON MOUSE
		if event.type == MOUSEBUTTONUP:
			write = False
			# SORT THE X_COORD AND Y_COORD
			x_coord = sorted(x_coord)
			y_coord = sorted(y_coord)
			# POINTS AROUND NUMBER (X1,Y1)- TOP LEFT, (X2,Y2)- BOTTOM RIGHT, (X3,Y3)-WIDTH, HEIGHT OF BOX
			x1 = max(x_coord[0] - bound,0)
			y1 = max(y_coord[0] - bound,0)
			x2 = min(x_coord[-1] + bound,screen_width)
			y2 = min(y_coord[-1] + bound,screen_height)
			x3 = x2 - x1
			y3 = y2 - y1
			# MAKE LIST EMPTY FOR NEW NUMBER TO INPUT
			x_coord = []
			y_coord = []
			# MAKE ARRAY OF THAT NUMBER
			array_of_num = pygame.surfarray.array2d(canvas)[x1:x2,y1:y2]
			image = np.array(array_of_num).T.astype("float32")

			# CODE FOR SAVING IMAGE
			if save_image:
				# ADD PATH	
				path = "images/"
				#USING OPENCV TO SAVE FILE
				cv2.imwrite(path + "image-{%d}.jpg" % n_image,image)
				#INCREASING NUMBER OF IMAGES
				n_image+=1

			# CODE FOR PREDICTION
			img = cv2.resize(image, (28,28))
			img = img.reshape((1,28,28,1))
			y_pred = model.predict(img)
			number = (np.argmax(y_pred,axis = 1))
			# CODE TO SHOW NUMBER ON SCREEN
			num = LABEL[number[0]]
			text_on_screen = font.render(num + " : " + str(number[0]) ,True,red,green)
			text_rect = text_on_screen.get_rect()
			text_rect.left = x1
			text_rect.bottom = y1
			canvas.blit(text_on_screen,text_rect)
			pygame.draw.rect(canvas,green,(x1,y1,x3,y3),3)
		
	pygame.display.update()

	if event.type == KEYDOWN:
		if event.unicode == "q":
			canvas.fill(black)
pygame.quit()
quit()


