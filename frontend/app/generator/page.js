'use client'

import React from "react";
import Editor from "./components/Editor";
import Image from "next/image";
import { useRouter } from "next/navigation";

const page = () => {

  const router = useRouter();

  return (
    <div className="bg-[#1E1E1E] min-h-screen flex">
      <div className="bg-[#222223] w-[5%] flex flex-col items-center h-[10vh] justify-end">
        <div className="text-white  w-[2.5em] h-[2.5em] relative cursor-pointer" onClick={() => router.push("/")}>
          <Image src={"/logo.svg"}  alt="company logo"
              fill
              style={{ height: "100%", width: "100%", objectFit: "cover" }}/>
        </div>
      </div>

      <div className="flex-1 flex">
        <div className="flex-1">
          <Editor />
        </div>
      </div>
    </div>
  );
};

export default page;
