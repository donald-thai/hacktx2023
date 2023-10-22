"use client";

import { useState, useCallback, useEffect } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";

const Editor = ({ generator, id }) => {
  const [value, setValue] = useState("print('hello world!')");
  const onChange = (val) => {
    setValue(val)
  };

  useEffect(() => {
    console.log(value)
  }, [value])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="bg-[#1E1F1F] h-[10vh] items-end flex">
        <div className="bg-[#28292A] w-fit py-2 px-5 border-solid border-b-2 border-[#39F4F9]">
          <p className="text-[#A0A0A0]">Example.py</p>
        </div>
      </div>

      <CodeMirror
        value={value}
        className="h-[90vh] overflow-auto"
        extensions={[python()]}
        onChange={onChange}
        theme={vscodeDark}
        editable={generator ? false : true}
      />
    </div>
  );
};

export default Editor;
