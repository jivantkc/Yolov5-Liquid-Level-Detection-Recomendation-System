<h1>Flask App to predict % of drinks in a glass & Recommend food or drinks</h1>
<h2>Flask, Yolo-v5, Content Based & Collaborative Recommendation system</h2>
<p>This is our final capstone project for Computer Vision and Recommendation System. Here we have trained YOLOV5 Model to detect our custome objects.We trained our model to detect top of glass, level of drink and base of drink in a glass. Once these objects in image are detected we calculate the % by dividing the distance between <b>base to level</b>/<b>base to top</b>. If % of drink is more than 50% then system recommends food using collaborative filtering and if its below 50% it recommends beers using content based filtering.</p>
<img src="drinksless.png">
<img src="drinksmore.png">
<hr>
<h2>Yolo V5 Image Dectection</h2>
<p>We took more than 500 picture of 3 Different types of drinks in 4 different types of glasses.
<img src="glassnliquid.jpg">
We labeled (Top, Level, Base) on each images using labelimg. Once images are ready we trained yolov5 on google collaboratory. Then we downloaded the trained weight to use on our flask app.
</p>


<h2>Food Recommendation System</h2>
<p>Food Recommendation system is using collaborative recommendation system. We have csv file in static folder in app containing the cutomer order table of more than 500 users. with different cafe name and different food orders. Technically the system finds distance between each users and recommends current user with the nearest distance to get the similar users. Then the foods that the current user has not ordered will be displayed but the similar user has already ordered that food.
<img src="drinksmore.png">
 
</p>


<h2>Beer Recommendation System</h2>
<p>Beer data of more than 5000 beers has been downloaded from kagle. Then we filtered out to just 500 beers. Each beer has its ingredients details. The system finds similarities between each products and recommends similar beer that you are having now.
<img src="drinksless.png">
</p>
