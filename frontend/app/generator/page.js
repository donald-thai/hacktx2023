import React from "react";
import Editor from "./components/Editor";

const page = () => {
  return (
    <div className="bg-[#1E1E1E] min-h-screen flex">
      <div className="bg-[#222223] w-[5%] flex flex-col items-center pt-10">
        <div className="text-white  w-fit">H</div>
      </div>

      <div className="flex-1 flex">
        <div className='flex-1'>
          <Editor />
        </div>

        {/* <div className='flex-1'>
          <Editor 
            generator={true}
          />
        </div> */}
      </div>
    </div>
  );
};

export default page;
