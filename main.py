import cv2
import sys
import predict
import face_detection as fd

if __name__ == '__main__':
	"""
	首先从图片中检测出人脸，然后进行评分
	"""
	for i in sys.argv:
		if i.find('.jpg') != -1:
			img = cv2.imread(i)

			shape = img.shape
			if shape[0] > 1000 or shape[1] > 1000:
				print(shape)
				size = (int(shape[0] / 2), int(shape[1] / 2))
				img = cv2.resize(img, size)
			# show(img)

			img_drawed = fd.draw_faces(img)
			font = cv2.FONT_HERSHEY_SIMPLEX
			faces, coordinates = fd.get_face_image(img)
			for i in range(len(faces)):
				# score = predict.predict_cv_img(faces[i])
				cv2.putText(img_drawed, str(predict.predict_cv_img(faces[i])), coordinates[i], font, 0.8, (255, 0, 0), 2)
			fd.show(img_drawed)
