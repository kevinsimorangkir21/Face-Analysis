import cv2
import dlib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
detector = dlib.get_frontal_face_detector()


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite('static/result.jpg', img)
            return render_template('index.html', num_faces=len(faces), image='static/result.jpg')

    return render_template('index.html', num_faces=None, image=None)


if __name__ == '__main__':
    app.run(debug=True)
