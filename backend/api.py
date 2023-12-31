from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import openai
openai.api_key = 'YOUR_API_KEY'

app = Flask(__name__)
api = Api(app)
CORS(app)

class CodexComplete(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        incomplete_code = data.get("code")
        if not incomplete_code:
            return {"message": "Code string not provided."}, 400
        
        completed_code = self.complete_with_codex(incomplete_code)
        print(completed_code)
        return jsonify({"completed_code": completed_code})

    

    def complete_with_codex(self, code):
        response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Complete the code that ends on this line. Respond only with the code that follows this and nothing more. Only autocomplete 1 line:\n" + code,
        )
        # Take the original length and the next 10 words
        response = response.choices[0].text
        return response


api.add_resource(CodexComplete, "/complete")

if __name__ == "__main__":
    app.run(debug=True, port=2000)
