from flask import Flask, request, jsonify
from sklearn import logger
from recommendation import get_recommendations
from suggestions import get_suggestions

app = Flask(__name__)

# Existing '/recommend' endpoint for user input
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json.get('user_input', '')
        level = request.json.get('level', '')  # Get the 'level' parameter from the request
        platform = request.json.get('platform', '')  # Get the 'platform' parameter from the request

        print("user request: ", user_input)
        print("level: ", level)
        print("platform: ", platform)

        # Pass the 'level' and 'platform' parameters to the get_recommendations function
        recommended_courses = get_recommendations(user_input, level, platform)
        print("courses: ", recommended_courses)

        return jsonify(recommended_courses.to_dict(orient='records'))

    except Exception as e:
        logger.error("An error occurred in get_recommendations: %s", e)
        return jsonify(error=str(e)), 500

# Updated '/suggestions' endpoint for receiving keywords from the app
@app.route('/suggestions', methods=['POST'])
def suggestions():
    try:
        search_words = request.json.get('search_words', [])
        print("user search words on login: ", search_words)

        # Implement the logic to fetch recommendations based on search words
        recommended_courses = get_suggestions(search_words)
        print("courses based on search words: ", recommended_courses)

        return jsonify(recommended_courses.to_dict(orient='records'))

    except Exception as e:
        logger.error("An error occurred in suggestions: %s", e)
        return jsonify(error=str(e)), 500

@app.route('/dummy', methods=['GET'])
def dummy():
    return jsonify(message="This is a dummy endpoint for testing purposes.")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
