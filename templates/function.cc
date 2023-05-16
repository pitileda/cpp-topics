#include <iostream>
#include <tuple>
#include <string_view>
#include <type_traits>

void do_smth(bool b, int i, std::string_view sv){
	std::cout << "do somethig with " << b
				<< ", "
				<< i
				<< ", "
				<< sv << std::endl; 
}

template<typename... Ts>
void wrapper(Ts&& ... args)
{
	static_assert(sizeof...(args) == 3, "Invalid number of arguments");

	std::tuple<bool, bool, bool> is_arg_set;
	std::tuple<bool, int, std::string_view> f_args;

	auto bind_arg = [&](auto &&arg){
		using arg_type = typename std::remove_reference<decltype(arg)>::type;

		if constexpr (std::is_same<arg_type, bool>::value){
			std::get<0>(is_arg_set) = true;
			std::get<0>(f_args) = std::forward<decltype(arg)>(arg);
		} else if constexpr (std::is_same<arg_type, int>::value){
			std::get<1>(is_arg_set) = true;
			std::get<1>(f_args) = std::forward<decltype(arg)>(arg);
		} else if constexpr (std::is_same<arg_type, std::string_view>::value){
			std::get<2>(is_arg_set) = true;
			std::get<2>(f_args) = std::forward<decltype(arg)>(arg);
		} 
		// else {
		// 	static_assert(std::is_null_pointer<decltype(arg)>::value, "Invalid argument type");
		// }
	};

	(bind_arg(std::forward<decltype(args)>(args)), ...);

	// if (!std::apply([](auto... is_arg_bound) {
	// 	return (is_arg_bound && ...);
	// }, is_arg_bound)) {
	// 	std::cerr << "Invalid arguments" << std::endl;
	// 	return;
	// }

	std::apply(do_smth, f_args);
}

int main(){
	using std::cout;
	using std::endl;
	wrapper(true, 12, "Hello");
	wrapper(12, true, "Hello");
	wrapper("Hello", true, 12);
	// cout << wrapper(true, 12, "Hello") << endl;
	return 0;
}