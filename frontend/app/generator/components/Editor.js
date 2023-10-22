"use client";

import { useState, useCallback, useEffect } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";

const Editor = ({ generator, id }) => {
  const suggestion = "finish this line";
  const [value, setValue] = useState("print('hello world!')");
  const [hasSuggestion, setHasSuggestion] = useState(false);

  const moveSuggestion = () => {
    const cursor = document.getElementsByClassName("cm-cursor-primary")[0];
    const suggestion = document.getElementById("suggest");
    suggestion.style.top = cursor.style.top;
    suggestion.style.left = cursor.style.left;
  };

  const removeSuggestion = () => {
    const suggestion = document.getElementById("suggest");
    suggestion.style.top = 0;
    suggestion.style.left = "10000px";
  };

  const onChange = (val) => {
    console.log(val);
    setValue(val);
    removeSuggestion();
    //setHasSuggestion(false);
  };

  useEffect(() => {
    const timeout = setTimeout(() => {
      console.log("stopped");
      setHasSuggestion(true);
      moveSuggestion();
    }, 500);
    return () => clearTimeout(timeout);
  }, [value]);

  useEffect(() => {
    document.addEventListener("keydown", (e) => {
      if (e.key == "Tab") {
        e.preventDefault();
        setValue(value + suggestion);
      }
      removeSuggestion();
    });
  });

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      <div className="bg-[#1E1F1F] h-[10vh] items-end flex">
        <div className="bg-[#28292A] w-fit py-2 px-5 border-solid border-b-2 border-[#39F4F9]">
          <p className="text-[#A0A0A0]">Example.py</p>
        </div>
      </div>
      <div className="relative">
        <CodeMirror
          value={value}
          className="h-[90vh] overflow-auto"
          extensions={[python()]}
          onChange={(e) => onChange(e)}
          theme={vscodeDark}
          editable={generator ? false : true}
          indentWithTab={false}
        />
        <div
          className="h-[19px] absolute px-2 text-slate-500 bg-slate-700 flex flex-row items-center justify-center"
          id="suggest"
        >
          <div>Suggestion: {suggestion}</div>
        </div>
      </div>
    </div>
  );
};

export default Editor;
