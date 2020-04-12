#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:39:41 2020

@author: rajan
"""
import os
from flask import Flask, flash, request, redirect, url_for, render_template, make_response
from werkzeug.utils import secure_filename
from main import style_image


#UPLOAD_FOLDER = '/home/rajan/Documents/style-transfer'
UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

filename= ''
bgfile = ''
stylefile = ''
#messages = False
#dfile = ''
#dfile = os.path.join(app.config['UPLOAD_FOLDER'], dfile)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/download', methods=['GET', 'POST'])
def download_Image(): 
    print('in download - '+dfile)
            
    return render_template('download.html', dfile=dfile)



@app.route('/style', methods=['GET', 'POST'])
def choose_StyleImage():
    global stylefile
    global filename
    global bgfile
    global messages
    global dfile
    if request.method == 'POST':
        print('in style- - '+filename)
        print('in style - '+bgfile)
        stylefile = request.form['stylefile']
        print('in style - '+stylefile)
        img = style_image(UPLOAD_FOLDER,filename, bgfile, stylefile)
        print(img)
        #messages = True
        dfile = img
     #   print('image type',+type(dfile))
        #img.show()
        
        #return render_template('download.html', dfile=dfile)
        return redirect(url_for('download_Image'))
    
    return render_template('style_images.html')

        
@app.route('/choose', methods=['GET', 'POST'])
def choose_bgImage():
    global bgfile
    global filename
    if request.method == 'POST':
        print('in choose - '+filename)
        bgfile = request.form['bgfile']
        print('in choose - '+bgfile)
        return redirect(url_for('choose_StyleImage'))
       
    return render_template('Select_background.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global filename
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER']+"/raw", filename))
            print(filename)
            #choose_bgImage(filename)
            
            
            return redirect(url_for('choose_bgImage'))
        
    return render_template('index.html')


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER']+"/raw",
                               filename)
    
if __name__ == "__main__":
    app.run()