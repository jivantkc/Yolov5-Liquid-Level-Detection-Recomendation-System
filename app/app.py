"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""



import argparse
import io
import os
from PIL import Image 
import PIL 
from flask import jsonify
from flask import Flask, request, render_template, session, redirect,jsonify,send_file
import sqlite3
from flask_sqlalchemy import SQLAlchemy
import boto3
import torch
import math
import base64
import pandas as pd
import sys
import cv2
import numpy as np
from flask_cors import CORS
import random
from datetime import datetime
import urllib.request
import json
from werkzeug.utils import secure_filename
import tempfile
import re

from scipy.spatial.distance import hamming 

# import required packages for beer recommendation
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
from sklearn.preprocessing import StandardScaler
# from keras.preprocessing import image



app = Flask(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///site.db"
db=SQLAlchemy(app)

class Images(db.Model):
    id=db.Column(db.Integer, primary_key=True )
    image_file=db.Column(db.String(220), nullable=False, default="default.jpg")
    converted_image=db.Column(db.String(220), nullable=False, default="converteddefault.jpg")
    country=db.Column(db.String(100), nullable=False,default="HK")
    city=db.Column(db.String(120), nullable=False,default="HK")
    level=db.Column(db.Integer, nullable=False, default=0)
    joke_id=db.Column(db.Integer, nullable=False,default=0)
    date_posted=db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    def __repr__(self):
        return f"Images ('{self.image_file}','{self.converted_image}','{self.country}','{self.city}','{self.level}','{self.joke_id}','{self.date_posted}'"



UPLOAD_FOLDER = "captured"
BUCKET = "glassvision-2021"

jokes=pd.read_csv('app/static/puns.csv')
beerData=pd.read_csv("app/static/beer_data_set.csv")
menus=pd.read_excel("app/static/menus.xlsx")
menus["Food Name"]=menus["Food Name"].apply(lambda x: re.sub("[^A-Za-z ]", '', x))
menus["Food Name"]=menus["Food Name"].str.title()
menus['menudf_index'] = menus.index

foodOrderData=pd.read_csv("app/static/ordertableuse.csv")
foodOrderData.drop("Unnamed: 0", axis=1, inplace=True)



def recommendjoke(x):
    y=jokes["puns"][x]
    return y


prediction={"id":0, "level":0,"joke_id":0,"joke":"","recommendation":[],"user":"", "beer_key":"", "currentbeer":[],"cb_alcohol":[] }


prediction["joke_id"]=random.randint(0,100)
prediction["joke"]=recommendjoke(prediction["joke_id"])

# Collaborative Food Recommender:
def foodrecommendationapp(select_shop):
    select_shop=select_shop
    new_order_table=pd.merge(foodOrderData, menus, on=['menudf_index'])
    new_order_table.drop("menudf_index", inplace=True, axis=1)
    shopaa=new_order_table[(new_order_table["Shop"]==select_shop)]
    newtd2=shopaa.groupby(["user","drinks_type", "Food Name"]).count()['Price'].reset_index()
    newtd2=newtd2.rename(columns={"Price": "order_count"})

    user_list=newtd2.user.unique().tolist()
    user1=user_list[random.randint(0, len(user_list))]
    select_user=user1
    prediction["user"]=select_user

    # Now lets create the user-item-rating matrix
    userOrderMatrix=pd.pivot_table(newtd2, values='order_count',index=['user'], columns=['Food Name',])


    # To Find the k nearest neighbours of active user first find the distance of active user to all other users
    def nearestneighbours(user,K):
        # create a user df that contains all users except active user
        allUsers = pd.DataFrame(userOrderMatrix.index)
        allUsers = allUsers[allUsers.user!=user]
        # Add a column to this df which contains distance of active user to each user
        allUsers["distance"] = allUsers["user"].apply(lambda x: hamming(userOrderMatrix.loc[user],userOrderMatrix.loc[x]))
        KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["user"][:K]
        return KnearestUsers


    def topN(user,N=5):
        KnearestUsers = nearestneighbours(user,N)
        # get the orders given by nearest neighbours
        NNOrders = userOrderMatrix[userOrderMatrix.index.isin(KnearestUsers)]
        # Find the average rating of each book rated by nearest neighbours
        avgOrder = NNOrders.apply(np.nanmean).dropna()
        # drop the books already read by active user
        foodsAlreadyEaten = userOrderMatrix.loc[user].dropna().index
        avgOrder = avgOrder[~avgOrder.index.isin(foodsAlreadyEaten)]
        topOrders = avgOrder.sort_values(ascending=False).index[:N]
        return topOrders

    food_recommendations=[]
    for index in topN(select_user):
        food_recommendations.append(index)
    
    return food_recommendations

# BEER RECOMMENDATION USING CONTENT BASED FILTERING

def beerecommendationapp():

    beermenus=beerData.set_index('key')
    #Getting less alcoholic beers
    beermenus=beermenus[(beermenus["ABV"]<5.1)&(beermenus["Max IBU"]<25)]
    beerinfo = beermenus.select_dtypes(np.number)
    list_to_drop=["Style Key",	"ABV",	"Ave Rating",	"Min IBU",	"Max IBU",	"Astringency",]
    newbeerinfo=beerinfo.copy()
    newbeerinfo.drop(list_to_drop, inplace=True, axis=1)
    newbeerinfo.drop(["Salty", "Body"], inplace=True, axis=1)

    beerdetails=beermenus.select_dtypes("object")
    # beercounts=beerdetails.groupby("Style").count()["Name"].reset_index()


    # standardize nutrition data by columns
    # standardize nutrition data by columns
    std = StandardScaler()
    df_scaled = std.fit_transform(newbeerinfo)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = newbeerinfo.columns
    df_scaled.index = newbeerinfo.index
    df_scaled.head()


    # show recipe id, recipe name and image of selected recipe
    def show_beer(beer_key):
        try:
            beer=beerdetails[beerdetails.index==beer_key]
            name=beer.Name.values[0]
            style=beer.Style.values[0]
            brewery=beer.Brewery.values[0]
            description = beer.Description.values[0]
            # print recipe name for this recipe_id
            return {"Name":name, "Style":style, "Brewery":brewery, "Description":description}
             
        except:
            print("img not found")


    """
    Beer Recommender based on different distance calculation approaches

    df_normalized: normalized beer data
    distance_method: distance calculation approach: euclidean
    recipe_id: find similar recipes based on the selected recipe
    N: Top N recipe(s)

    Return:
    1) nutrition data of selected recipe and Top N recommendation, 
    df_normalized_"""

    def beer_recommender(distance_method, key, N):
        # create dataframe used to store distances between recipes
        df_distance = pd.DataFrame(data=beerinfo.index)
        
        # remove rows where index is equal to the inputted recipe_id
        df_distance = df_distance[beerinfo.index != key]
        
        # add a distance column that states the inputted recipe's distance with every other recipe
        df_distance['distance'] = df_distance["key"].apply(lambda x: distance_method(df_scaled.loc[x],df_scaled.loc[key]))
        
        # sort the allRecipes by distance and take N closes number of rows to put in the TopNRecommendation as the recommendations
        df_distance.sort_values(by='distance',inplace=True)
        
        # for each recipe in TopNRecommendation, input to defined lists
        
        # return dataframe with the inputted recipe and the recommended recipe's normalized nutritional values
        return(df_distance.head(N))


    # print(nutrition_recommender(euclidean, 233654, 3))
    distance_method=cosine
     

    ##print current and recommendatiosn with nutrition values and pictures

    # print("Current Beer:")
    # print("\n")
    key=beerdetails.index.tolist()
    newkey=random.randint(0, len(key))
    current_key=key[newkey]

    prediction["beer_key"]=current_key

    # print(beerinfo.loc[[current_key]])

    show_beer(current_key)
   
    
    # # print("\n")
    # print("Recommended Beers: ")
    x=beer_recommender(distance_method, current_key, 5)

    beer_recommendation=[]
    prediction["currentbeer"]= show_beer(current_key) 
    prediction["cb_alcohol"]= newbeerinfo.loc[[current_key]].to_dict("list")

    for i in range(len(x.index)):
        beer_recommendation.append(show_beer(x['key'].values[i]))

    return beer_recommendation

        # # print(beerinfo.loc[[x['key'].values[i]]])
        # prediction["recommendation"].clear()
        # prediction["recommendation"].append(show_beer(x['key'].values[i]))




 




@app.route("/", methods=["GET", "POST"])

def jpredict():
    if request.method == "POST":
        select_shop=request.form.get('cafename')
        print(select_shop)
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        model = torch.hub.load('app/', 'custom', path='app/runs/train/yolov5s_level/weights/last.pt', source='local')
        # Inference Settings
        # Inference settings such as confidence threshold, NMS IoU threshold, and classes filter are model attributes, and can be modified by:

        model.conf = 0.30  # confidence threshold (0-1)
        model.iou = 0.45  # NMS IoU threshold (0-1)
        # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs



        results = model(img, size=1200)
        results.display(save=True, save_dir="app/static/converted")
 
        # results.save()  # or .show()
        # ans2=results.xyxy[0]  # img1 predictions (tensor)
        # ans2=results.pandas().xyxy[0]  # img1
        x=results.pandas().xyxy[0]
        # ans2.to_html(header="true", table_id="table")
        # Image(filename='runs/detect/exp/IMG_0978.JPG', width=400) 
        if "Level" in x.name.values and "Top" in x.name.values and "Base" in x.name.values:
            top=x[x.name=="Top"]
            top=top[top.confidence==top.confidence.max()]
            top=(top.ymin+top.ymax)/2

            level=x[x.name=="Level"]
            level=level[level.confidence==level.confidence.max()]
            level=(level.ymin+level.ymax)/2
            level


            base=x[x.name=="Base"]
            base=base[base.confidence==base.confidence.max()]
            base=base.ymin
         

            size=base.values[0]-top.values[0]
            volume=base.values[0]-level.values[0]
            levels=round(volume/size*100)
            if levels<1:
                levels=0
            prediction["level"]=levels

            if levels>49:
                prediction["recommendation"]=foodrecommendationapp(select_shop)
            else:
                prediction["recommendation"]=beerecommendationapp()
            return render_template("success.html", level=prediction)

        else:
            levels=0
            prediction["level"]=levels
            prediction["recommendation"]=beerecommendationapp()
            return render_template("success.html", level=prediction)

            
        

        # return render_template('success.html',  tables=[ans.to_html(classes='data')], titles=ans.columns.values)

    return render_template("index.html")



# @app.route("/api", methods=["GET"])

# def get():
#     return jsonify({"Prediction":prediction})

"""


@app.route("/api/", methods=["POST","GET"])

def predictions():
     
    # with urllib.request.urlopen("https://geolocation-db.com/json") as url:
    #         data = json.loads(url.read().decode())
    #         prediction["country"]=data["country_name"]
    #         prediction["city"]=data["city"]
            
    

    if request.method == "POST":
        prediction["joke"]=recommendjoke(prediction["joke_id"])
        filuz=request.data
        x=str(filuz)
        ff=x.split(",")[1]
        imgdata = base64.b64decode(ff)
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # print (timestr)

        # filename=timestr+".jpg"
        filename="captured.jpg"
        
        fp = tempfile.TemporaryFile()
        fp.write(imgdata)
        img=fp.read()
        fp.close()

        
        with open(filename, 'wb') as f:
            f.write(imgdata)
        
        # im1 = Image.open(filename)
        # im1.save(os.path.join(UPLOAD_FOLDER, secure_filename(filename)))


        # objectname = "captured/"+filename
        # s3_client = boto3.client('s3')
        # response = s3_client.upload_file(filename, BUCKET, objectname)
        
        


        # img = Image.open(io.BytesIO(filename))
        # img=filename
        model = torch.hub.load('app/', 'custom', path='app/runs/train/yolov5s_level/weights/last.pt', source='local')
        # Inference Settings
        # Inference settings such as confidence threshold, NMS IoU threshold, and classes filter are model attributes, and can be modified by:

        model.conf = 0.30  # confidence threshold (0-1)
        model.iou = 0.45  # NMS IoU threshold (0-1)
        # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs



        results = model(img, size=1200)
        results.display(save=True, save_dir="/Users/jivantkhatri/ML/react/level-tracker/public/converted")

        # objectname1 = "app/static/converted"+filename
        # s3_client1 = boto3.client('s3')
        # converted1 = s3_client1.upload_file(filename, BUCKET, objectname1)

 
        # results.save()  # or .show()
        # ans2=results.xyxy[0]  # img1 predictions (tensor)
        # ans2=results.pandas().xyxy[0]  # img1
        x=results.pandas().xyxy[0]
        # ans2.to_html(header="true", table_id="table")
        # Image(filename='runs/detect/exp/IMG_0978.JPG', width=400) 



        # prediction["imagefile"]="https://glassvision-2021.s3.us-west-2.amazonaws.com/"+objectname
        # prediction["convertedimage"]="https://glassvision-2021.s3.us-west-2.amazonaws.com/"+objectname1


        if "Level" in x.name.values and "Top" in x.name.values and "Base" in x.name.values:
            jk=recommendjoke(prediction["joke_id"])
            prediction["joke"]=jk
            top=x[x.name=="Top"]
            top=top[top.confidence==top.confidence.max()]
            top=(top.ymin+top.ymax)/2

            level=x[x.name=="Level"]
            level=level[level.confidence==level.confidence.max()]
            level=(level.ymin+level.ymax)/2
            level

            base=x[x.name=="Base"]
            base=base[base.confidence==base.confidence.max()]
            base=base.ymin
         
            size=base.values[0]-top.values[0]
            volume=base.values[0]-level.values[0]
            levels=round(volume/size*100)
            if levels<1:
                levels=0
            prediction["level"]=levels
           
            images1=Images(image_file=prediction["imagefile"],converted_image=prediction["convertedimage"],country=prediction["country"],city=prediction["city"],level=prediction["level"],joke_id=prediction["joke_id"])
            db.session.add(images1)
            db.session.commit()
            response= jsonify({"Created":prediction})
            response.headers.add('Access-Control-Allow-Origin', '*')
            print(response.data)
            return response
            
            

        else:
            jk=recommendjoke(prediction["joke_id"])
            prediction["joke"]=jk
            levels=0
            prediction["level"]=levels

            images2=Images(image_file=prediction["imagefile"],converted_image=prediction["convertedimage"],country=prediction["country"],city=prediction["city"],level=prediction["level"],joke_id=prediction["joke_id"])
            db.session.add(images2)
            db.session.commit()
            
            
            
    
        response= jsonify({"Created":prediction})
        response.headers.add('Access-Control-Allow-Origin', '*')
        print(response.data)
        return response
        
    else:
        jk=recommendjoke(prediction["joke_id"])
        prediction["joke"]=jk
        response= jsonify({"Prediction":prediction})
        response.headers.add('Access-Control-Allow-Origin', '*')
        print(response.data)
        return response

"""

if __name__ == "__main__":
    # app.run(host='0.0.0.0')
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    # model1 = torch.hub.load(
    #     "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    # ).autoshape()  # force_reload = recache latest code
    # model1.eval()
    app.run(host="0.0.0.0", port=args.port, ssl_context='adhoc')  # debug=True causes Restarting with stat
    
