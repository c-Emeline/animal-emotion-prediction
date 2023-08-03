from flask import Flask, render_template
from flask import jsonify
from flask import request
import commons as cm



def create_app():
    app = Flask(__name__)


    @app.route('/')
    def homepage():
        return render_template('homepage.html')


    @app.route('/predict/', methods=['POST','GET'])
    def predict():
        jsonify(
            class_id="class id a retourner",
            class_name="nom de la classe à retourner"
        )
        return render_template('result.html')


    @app.route('/image/', methods=['POST'])
    def image():
        return render_template('index.html')


    @app.route('/text/', methods=['POST'])
    def text():
        return render_template('text.html')


    @app.route('/predictiontxt/', methods=['POST','GET'])
    def predict_txt():
        if request.method == 'POST':
            txt = request.form.get('txt_input')
            label, score = cm.get_prediction_txt_roberta(txt)
            return render_template('result_txt.html', emotion_label = label, emotion_score = score)


    @app.route('/predictionimg/', methods = ['GET', 'POST'])
    def predict_img():
        if request.method == 'POST':
            f = request.files['file']
            if f.filename!="" :
                f.save(f.filename)

                if request.form.get("model") == "Densenet121" :
                    classname, classid = cm.get_prediction_img_121(f.filename)

                if request.form.get("model") == "Densenet201" :
                    classname, classid = cm.get_prediction_img_201(f.filename)

                return render_template('result.html', class_name=classname, class_id=classid.item())

            else :
                return render_template('index.html')


    return app