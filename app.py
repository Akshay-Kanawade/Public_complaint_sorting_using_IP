from flask import Flask, render_template, request
from tools.prediction import test_on_img
from tools.json_updater import update_json, get_json_data
from config import CLASSES
import numpy as np
import cv2
import time
from datetime import date

app = Flask("__name__")


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def my_form_post():
    name = request.form['name']
    mobile = request.form['mobile']
    address = request.form['address']
    description = request.form['description']
    # get currernt time and date
    current_time = time.time()
    date_today = str(date.today())
    # read image
    image = request.files['image'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img_src = f"data/complaint_images/{current_time}.jpg"
    cv2.imwrite(img_src, img)
    plot, prediction, pred = test_on_img(img_src)
    s = [str(i) for i in prediction]
    a = int("".join(s))
    print("Predicted Image class is: ", CLASSES[a])
    user_data = {
        'name': name,
        'mobile': mobile,
        'address': address,
        'description': description,
        'date': date_today,
        'image_path': img_src,
        'class': CLASSES[a]
    }

    update_json(user_data=user_data)

    result = f"Your Complaint Registered in {user_data['class']} Successfully..."

    return str(result)


@app.route("/admin")
def admin():
    json_data = get_json_data()
    electricity = json_data['electricity_department']
    garbage = json_data['garbage_department']
    street = json_data['street_department']
    return render_template("admin2.html", electricity_list=electricity, garbage_list=garbage, street_list=street)


if __name__ == "__main__":
    app.run(debug="True")
