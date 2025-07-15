from flask import Flask, render_template, request
from dbconnect import connection
from flask_googlemaps import GoogleMaps
from flask_googlemaps import Map, icons

app = Flask(__name__)

GoogleMaps(app, key="AIzaSyBQp1EK4oaTlzdzAfjlNX76_HHDv2khEYQ")

@app.route("/")
def homepage():
    try:
        cur, conn = connection()
        cur = conn.cursor()
        cur.execute("SELECT user.name FROM user")
        user = cur.fetchall()
        user = [row[0] for row in user]
        cur.close()
        return render_template('index.html', users=user)

    except Exception as e:
        return('Connection error')



@app.route("/map")
def mapview():

    cur, conn = connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bin")
    binn = cur.fetchall()
    cur.close()
    bin_id = [row[0] for row in binn]
    bin_type = [row[1] for row in binn]
    bin_capacity = [row[2] for row in binn]
    cur.close()

    icon = []

    for item in bin_capacity:
        if item == 0:
            icon.append("https://imagizer.imageshack.com/v2/100x75q90/922/5cNWwn.png")
        elif item <= 25:
            icon.append("https://imagizer.imageshack.com/v2/100x75q90/924/caSf6J.png")
        elif item <= 50:
            icon.append("https://imagizer.imageshack.com/v2/100x75q90/922/UT703T.png")
        elif item <= 75:
            icon.append("https://imagizer.imageshack.com/v2/100x75q90/923/uQj3zL.png")
        elif item <= 100:
            icon.append("https://imagizer.imageshack.com/v2/100x75q90/922/fNtiZk.png")

    

    sndmap = Map(
            identifier="sndmap",
            lat=43.715993,
            lng=10.3960694,

            markers=[
                {
                    'icon': icon[0],
                    'lat': 43.716248,
                    'lng': 10.402882,
                    'infobox':"<p><b> ID: </b>" +str(bin_id[0])+ "</p> <p><b> Type: </b> " +str(bin_type[0])+ "</p> <p><b> Capacity: </b> " +str(bin_capacity[0])+ "%</p>"
                },
                {
                    'icon': icon[1],
                    'lat': 43.718326,
                    'lng': 10.398201,
                    'infobox':"<p><b> ID: </b> " +str(bin_id[1])+ "</p> <p><b> Type: </b>  " +str(bin_type[1])+ "</p> <p><b> Capacity: </b> " +str(bin_capacity[1])+ "%</p>"
                },
                {
                    'icon': icon[2],
                    'lat':43.719660 ,
                    'lng':10.403909 ,
                    'infobox':"<p><b> ID: </b> " +str(bin_id[2])+ "</p> <p><b> Type: </b>  " +str(bin_type[2])+ "</p> <p><b> Capacity: </b> " +str(bin_capacity[2])+ "%</p>"
                },
                {
                    'icon': icon[3],
                    'lat':43.716660,
                    'lng':10.403909,
                    'infobox':"<p><b> ID: </b> " +str(bin_id[3])+ "</p> <p><b> Type: </b>  " +str(bin_type[3])+ "</p> <p><b> Capacity: </b> " +str(bin_capacity[3])+ "%</p>"
                },
                {
                    'icon': icon[4],
                    'lat':43.718481 , 
                    'lng':10.408117 , 
                    'infobox':"<p><b> ID: </b> " +str(bin_id[4])+ "</p> <p><b> Type: </b>  " +str(bin_type[4])+ "</p> <p><b> Capacity: </b> " +str(bin_capacity[4])+ "%</p>"
                },
                ],

            style = "height: 700px; width:1200px; margin:0",
            zoom = "15"
            )
    return render_template('map.html', sndmap=sndmap)


@app.route("/bins", methods= ['GET', 'POST'])
def bin():
    if request.method == 'POST':
        try:
            cur, conn = connection()
            cur = conn.cursor()
            cur.execute("SELECT type FROM bin")
            bin_type = cur.fetchall()
            bin_type = [row[0] for row in bin_type]
            cur.execute("SELECT capacity FROM bin")
            capacity = cur.fetchall()
            capacity = [row[0] for row in capacity]
            cur.close()
            bar_labels = bin_type
            bar_values = capacity
            return render_template('bins.html', title='bins', max = 100, labels=bar_labels, values=bar_values)
    
        except Exception as e:
            return('Connection error')
    

@app.route("/user", methods=['GET','POST'])
def user_interaction():
    if request.method == 'POST':
        req = request.form
        name = req.get("utenti")
        try:
            cur, conn = connection()
            cur = conn.cursor()
            sql = "SELECT user.name, user_interaction.garbage_weight, user_interaction.timestamp FROM user INNER JOIN user_interaction ON user.card_id = user_interaction.user_id WHERE name = %s "
            name = (str(name), )
            cur.execute(sql, name)
            data = cur.fetchall()
            user = [row[0] for row in data]
            garbage = [row[1] for row in data]
            time = ["" + str(row[2].day) + "/" + str(row[2].month) + "/" + str(row[2].year) for row in data]
            bar_labels = time
            bar_values = garbage


            sql = "SELECT user.name, SUM(user_interaction.garbage_weight), bin.type FROM user JOIN user_interaction ON user.card_id = user_interaction.user_id JOIN bin ON user_interaction.bin_id = bin.id WHERE user.name = %s GROUP BY bin.type"
            cur.execute(sql, name)
            data = cur.fetchall()
            user2 = [row[0] for row in data]
            garbage2 = [row[1] for row in data]
            bin_type2 = [row[2] for row in data]
            bar_labels1 = bin_type2
            bar_values1 = garbage2

            return render_template('user.html', title=str(user[0]), max = 200, max2 = 1000, labels=bar_labels, values=bar_values, labels2=bar_labels1, values2=bar_values1)
    
        except Exception as e:
            return('Connection error')

@app.route("/point", methods=['GET','POST'])
def gamification():
    if request.method == 'POST':
        req = request.form
        name = req.get("utenti")
        try:
            cur, conn = connection()
            cur = conn.cursor()
            sql = "SELECT user.name, user.points FROM user WHERE name = %s "
            name = (str(name), )
            cur.execute(sql, name)
            data = cur.fetchall()
            user = [row[0] for row in data]
            points = [row[1] for row in data]
            bar_labels = user
            bar_values = points
            
            return render_template('point.html', title=str(user[0]), max = 5000, labels=bar_labels, values=bar_values)
    
        except Exception as e:
            return('Connection error')

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
