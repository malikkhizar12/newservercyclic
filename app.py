from flask import Flask, request, jsonify
from sklearn import logger
from recommendation import get_recommendations

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json.get('user_input', '')
        print("user request ", user_input)
        recommended_courses = get_recommendations(user_input)
        print("courses ", recommended_courses)
        
        return jsonify(recommended_courses.to_dict(orient='records'))
        
        
    except Exception as e:
        logger.error("An error occurred in get_recommendations: %s", e)
        return jsonify(error=str(e)),500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
