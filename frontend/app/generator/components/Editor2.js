"use client";

import { useState, useCallback, useEffect } from "react";

const Editor2 = ({ generator, id }) => {
  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      <div className="bg-[#1E1F1F] flex">
        <div className="bg-[#28292A] py-2 px-5 border-solid border-b-2 border-[#39F4F9]">
          <p className="text-[#A0A0A0]">Example.py</p>
        </div>
      </div>
      <div className="flex flex-grow"></div>
    </div>
  );
};

export default Editor2;
