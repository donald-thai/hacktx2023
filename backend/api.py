from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import openai
openai.api_key = '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'

app = Flask(__name__)
api = Api(app)

class CodexComplete(Resource):
    def post(self):
        data = request.get_json()
        incomplete_code = data.get("code")
        if not incomplete_code:
            return {"message": "Code string not provided."}, 400
        
        completed_code = self.complete_with_codex(incomplete_code)

        return jsonify({"completed_code": completed_code})

    

    def complete_with_codex(self, code):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': "Complete the code that ends on this line. Respond only with the code that follows this and nothing more. Only autocomplete 1 line:\n" + code}],
        )
        # Take the original length and the next 10 words
        response = response.choices[0].message['content']
        return response


api.add_resource(CodexComplete, "/complete")

if __name__ == "__main__":
    app.run(debug=True, port=2000)
