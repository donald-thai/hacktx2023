"use client";

import React, { useRef, useEffect, useState } from "react";

import { EditorView, basicSetup } from "codemirror";
import { EditorState } from "@codemirror/state";
import { keymap } from "@codemirror/view";
import { python } from "@codemirror/lang-python";
import { vscodeDark } from "@uiw/codemirror-theme-vscode";

const Editor = () => {
  const editor = useRef();
  const [edit, setEdit] = useState();
  const [suggestion, setSuggestion] = useState("print(output)");
  const [value, setValue] = useState("");

  // Have we used a suggestion recently
  const [hasSuggestion, setHasSuggestion] = useState(true);

  const makeRequest = async () => {
    const rotated = encrypt(value);
    console.log("Send Encrypted Request with Data:", rotated);

    await fetch("http://localhost:2000/complete", {
      method: "POST",
      // mode: "cors", // no-cors, *cors, same-origin
      // cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
      // credentials: "same-origin", // include, *same-origin, omit
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      // redirect: "follow", // manual, *follow, error
      // referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify({ code: value }),
    })
      .then((res) => {
        console.log(res);
      })
      .catch((err) => console.log(err));
  };

  const encrypt = (text) => {
    let ans = "";
    for (let i = 0; i < text.length; i++) {
      ans += String.fromCharCode(text[i].charCodeAt(0) + 1);
    }
    return ans;
  };

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

  const onChange = EditorView.updateListener.of((v) => {
    setValue(v.state.doc.toString());
    removeSuggestion();
    setHasSuggestion(false);
  });

  useEffect(() => {
    if (!hasSuggestion) {
      let timeout;
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        let cursorLoc = edit.state.selection.main.from;
        let dataUpToCursor = edit.state.doc.toString().slice(0, cursorLoc);
        makeRequest();
        moveSuggestion();
      }, 700);
      return () => clearTimeout(timeout);
    }
  }, [value]);

  const addSuggestion = (edit) => {
    //console.log(edit.state.selection.main);
    let transaction = edit.state.update({
      changes: { from: edit.state.selection.main.from, insert: suggestion },
    });

    // At this point the view still shows the old state.
    edit.dispatch(transaction);

    setValue(transaction.state.doc.toString());
    removeSuggestion();
    setHasSuggestion(true);
  };

  useEffect(() => {
    const state = EditorState.create({
      doc: "",
      extensions: [
        basicSetup,
        keymap.of([
          {
            key: "Tab",
            run: (edit) => {
              addSuggestion(edit);
            },
          },
        ]),
        python(),
        vscodeDark,
        onChange,
      ],
    });

    const v = new EditorView({ state, parent: editor.current });
    setEdit(v);
    return () => {
      v.destroy();
    };
  }, []);

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      <div className="bg-[#1E1F1F] h-[10vh] items-end flex">
        <div className="bg-[#28292A] w-fit py-2 px-5 border-solid border-b-2 border-[#39F4F9]">
          <p className="text-[#A0A0A0]">Example.py</p>
        </div>
      </div>
      <div className="relative">
        <div ref={editor}></div>
        <div
          className="h-[19px] absolute text-slate-500 bg-slate-700 flex flex-row items-center justify-center left-[1000px]"
          id="suggest"
        >
          <div>{suggestion}</div>
        </div>
      </div>
    </div>
  );
};

export default Editor;
