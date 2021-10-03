import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import util as util
import pickle
import model as md
from pandas.plotting import parallel_coordinates

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/association')
def association_start():
    rule_list = util.rule_list()
    util.networkXplot(rule_list,10)
    plt.figure()
    plt.figure(figsize=(4, 8))
    plt.savefig('static/' + 'plot3.png', dpi=600, edgecolor="#04253a")
    return render_template('association.html')

@app.route('/associationN')
def association_startN():
    coords = util.parallelPlot()
    # Generate parallel coordinates plot
    plt.title('Parallel Coordinate Plot')
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.figure(figsize=(4,8))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.savefig('static/' + 'plot1.png', dpi=600, edgecolor="#04253a")
    plt.close()
    return render_template('association.html')

@app.route('/classification')
def classification_start():
    return render_template('classification.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    fields = []
    if request.method == "POST":
        fields_1 = request.form.get('gender')
        fields_2 = request.form.get('car')
        fields_3 = request.form.get('realty')
        fields_4 = request.form.get('child')
        fields_5 = request.form.get('income')
        fields_6 = request.form.get('income_type')
        fields_7 = request.form.get('educate')
        fields_8 = request.form.get('fam_stat')
        fields_9 = request.form.get('hou_type')
        fields_10 = request.form.get('age')
        fields_11 = request.form.get('experience')
        fields_12 = request.form.get('occ')
        fields_13 = request.form.get('fam_mem')
        fields_14 = request.form.get('month')

        prediction = md.predict_classification(fields_1, fields_2, fields_3, fields_4, fields_5, fields_6, fields_7, fields_8, fields_9, fields_10, fields_11, fields_12, fields_13, fields_14)

        print("prediction", prediction)
        if prediction == 0:
            output = '1-29 Days Past Dues'
        elif prediction == 1:
            output = 'More Than 30 Days Past Dues'
        elif prediction == 2:
            output = 'Paid Off'
        elif prediction == 3:
            output = 'No Loan'

        if prediction == 0 or prediction == 1:
            result = 'Customer Is Not Eligible for Credit Card'
        elif prediction == 2 or prediction == 3:
            result = 'Customer Is Eligible for Credit Card'

    return render_template('classification.html', prediction_text='Status : {}'.format(output), prediction_output= 'System Recommendation : {}'.format(result))


@app.route("/rule", methods=["GET", "POST"])
def pattern_analysis():
    listed = ""
    items_selected = []
    item_list = []
    if request.method == "POST":
        item_select1 = request.form.get('item_select1')
        item_select2 = request.form.get('item_select2')
        item_select3 = request.form.get('item_select3')

        if item_select1 != "0":
            items_selected.append(item_select1+ "    ")
            item_list.append(item_select1)
        if item_select2 != "0":
            items_selected.append(item_select2)
            item_list.append(item_select2)
        if item_select3 != "0":
            items_selected.append(item_select3)
            item_list.append(item_select3)

        item_selected = " , ".join([str(item) for item in items_selected])
        listed = util.recommend_product(item_list)
        rule_list = util.rule_list()
        util.networkPlotRule(rule_list, len(rule_list),item_list)

    return render_template('association.html', pattern1=listed,items=item_selected)


if __name__ == '__main__':
    app.run(debug=True)
