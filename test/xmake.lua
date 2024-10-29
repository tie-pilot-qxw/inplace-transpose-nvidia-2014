add_requires("libomp", {optional = true})


target("benchmark")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../inplace/cuda/*.cuh")
    add_files("../inplace/cuda/*.cu")
    add_headerfiles("../inplace/common/*.h")
    add_headerfiles("../inplace/common/*.cuh")
    add_files("../inplace/common/*.cu")
    add_files("../inplace/common/*.cpp")
    add_headerfiles("util/*.h")
    add_files("util/*.cu")
    add_files("benchmark.cu")
    add_includedirs("../inplace")
    add_cugencodes("native")

target("skinny")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../inplace/cuda/*.cuh")
    add_files("../inplace/cuda/*.cu")
    add_headerfiles("../inplace/common/*.h")
    add_headerfiles("../inplace/common/*.cuh")
    add_files("../inplace/common/*.cu")
    add_files("../inplace/common/*.cpp")
    add_headerfiles("util/*.h")
    add_files("util/*.cu")
    add_files("skinny.cu")
    add_includedirs("../inplace")
    add_cugencodes("native")

target("rotate")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../inplace/cuda/*.cuh")
    add_files("../inplace/cuda/*.cu")
    add_headerfiles("../inplace/common/*.h")
    add_headerfiles("../inplace/common/*.cuh")
    add_files("../inplace/common/*.cu")
    add_files("../inplace/common/*.cpp")
    add_headerfiles("util/*.h")
    add_files("util/*.cu")
    add_files("rotate.cu")
    add_includedirs("../inplace")
    add_cugencodes("native")

target("permute")
    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../inplace/cuda/*.cuh")
    add_files("../inplace/cuda/*.cu")
    add_headerfiles("../inplace/common/*.h")
    add_headerfiles("../inplace/common/*.cuh")
    add_files("../inplace/common/*.cu")
    add_files("../inplace/common/*.cpp")
    add_headerfiles("util/*.h")
    add_files("util/*.cu")
    add_files("permute.cu")
    add_includedirs("../inplace")
    add_cugencodes("native")


target("openmp")

    add_cxflags("-fopenmp")
    add_packages("libomp")

    set_languages(("c++20"))
    if is_mode("debug") then 
        set_symbols("debug")
    end
    add_headerfiles("../inplace/openmp/*.h")
    add_files("../inplace/openmp/*.cpp")
    add_headerfiles("../inplace/common/*.h")
    add_headerfiles("../inplace/common/*.cuh")
    add_files("../inplace/common/*.cu")
    add_files("../inplace/common/*.cpp")
    add_files("openmp.cpp")
    add_includedirs("../inplace")