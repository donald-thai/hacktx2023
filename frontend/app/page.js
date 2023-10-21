'use client'

import Image from "next/image";
import { useRouter } from 'next/navigation'

export default function Home() {

  const router = useRouter();

  return (
    <div className="bg-black h-screen w-sceen px-32 py-16">
      <div className=" text-white flex flex-col w-full h-full gap-3">
        <header className="font-bold">
          <h2 className="text-2xl mb-3">COMPANY NAME</h2>
          <div className="text-[7rem] leading-none flex gap-4">
            <div>
              <h1>//</h1>
            </div>
            <div>
              <h1>CODE WHAT</h1>
              <h1>YOU IMAGINE.</h1>
            </div>
          </div>
        </header>

        <div className="flex flex-1 gap-8">
          <div className="flex flex-col flex-1 justify-end gap-4">
            <h3 className="text-3xl font-semibold">AI driven coding assistance without the privacy issues.</h3>  
            <div className="py-3 px-6 bg-[#02484A] text-center w-fit font-bold cursor-pointer" onClick={() => router.push('/generator')}>GET STARTED</div>
          </div>

          <div className="w-1/2 max-h-full overflow-hidden relative">
            <Image
              src={"/grid.jpg"}
              alt="turquoise grid"
              fill
              style={{ height: "100%", width: "100%", objectFit: "cover" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
