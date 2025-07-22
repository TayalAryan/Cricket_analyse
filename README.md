# **Cricket Stance Analysis System**

This project provides a suite of tools to analyze the biomechanics of a cricket batsman's stance and shot execution from a video file. The system uses MediaPipe for pose detection and saves detailed, frame-by-frame analysis into a PostgreSQL database.

The workflow is split into two separate applications:

1. **Processor App**: A backend tool that finds unprocessed videos from a database queue, runs the analysis, and stores the results.  
2. **Viewer App**: A frontend dashboard for selecting a player and their processed videos to view the detailed analysis and frame-by-frame visualizations.

## **Prerequisites**

Before you begin, ensure you have the following installed:

* Python 3.9+  
* Git  
* PostgreSQL Server  
* pgAdmin 4 (or another PostgreSQL client)

## **Setup Instructions**

### **1\. Clone the Repository**

First, clone this branch of the repository to your local machine.

git clone https://github.com/TayalAryan/Cricket\_analyse.git  
cd Cricket\_analyse  
git checkout split-apps-workflow

### **2\. Install Dependencies**

Install all the required Python libraries using pip.

pip install streamlit opencv-python mediapipe psycopg2-binary

### **3\. Configure Database Credentials**

You must provide credentials for the application to connect to your PostgreSQL database.

1. In the main project folder, create a new folder named .streamlit.  
2. Inside the .streamlit folder, create a new file named secrets.toml.  
3. Add your database connection details to this file. **Replace the placeholder values with your actual credentials.**  
   \# .streamlit/secrets.toml

   \[postgres\]  
   host \= "localhost"  
   port \= 5432  
   dbname \= "cricket\_analysis"  
   user \= "your\_db\_user"  
   password \= "your\_db\_password"

## **Database Setup**

You need to create the database, a user, and the required tables.

1. **Open pgAdmin 4** and connect to your PostgreSQL server.  
2. **Create a User**: Right-click **Login/Group Roles** \> **Create**. Name the user (e.g., cricket\_app\_user) and give it a password and login privileges.  
3. **Create a Database**: Right-click **Databases** \> **Create**. Name the database cricket\_analysis and set the owner to the user you just created.  
4. **Create Tables**: Open the **Query Tool** for your new database and execute the entire SQL script found in database\_setup.sql (or copy it from the project documentation). This will create the players, videos, splitvideo\_dtls, and stance\_biomech\_base tables.

## **Running the Application: Workflow**

The system is designed to be run in a specific order.

### **Step 1: Populate the Processing Queue**

First, you must tell the system which videos to analyze.

1. Open the **Query Tool** in pgAdmin.  
2. For each video you want to process, you need to run three INSERT commands: one for the players table (if the player is new), one for the videos table, and one for the splitvideo\_dtls table (which acts as the processing queue).  
   **Example for a new player and video:**  
   \-- 1\. Add the player  
   INSERT INTO players (player\_id, name, batch) VALUES (1, 'Player Name', 'Batch A');

   \-- 2\. Link the video to the player  
   INSERT INTO videos (player\_id, title, url, video\_id)   
   VALUES (1, 'Video Title', 'C:\\\\path\\\\to\\\\your\\\\video.mp4', 101);

   \-- 3\. Add the video to the processing queue  
   INSERT INTO "splitvideo\_dtls" (splitvideo\_id, video\_id, vid\_path, vid\_fps, center\_sec)   
   VALUES (101, 101, 'C:\\\\path\\\\to\\\\your\\\\video.mp4', 30, 0);

### **Step 2: Run the Processor App**

This app works through the queue in the splitvideo\_dtls table one video at a time.

1. Open your command prompt, navigate to the project folder, and run:  
   streamlit run processor\_app.py

2. The app will find the first unprocessed video. Click the **"Process Video"** button.  
3. After processing, you must manually remove the video from the queue in pgAdmin before processing the next one.  
   DELETE FROM "splitvideo\_dtls" WHERE splitvideo\_id \= 101; \-- Use the correct ID

4. Refresh the processor app page to find and process the next video.

### **Step 3: Run the Viewer App**

This app lets you see the results of any videos that have been successfully processed.

1. Open a new command prompt, navigate to the project folder, and run:  
   streamlit run viewer\_app.py

2. Use the dropdown menus in the sidebar to select a player and then one of their processed videos to view the frame-by-frame analysis.