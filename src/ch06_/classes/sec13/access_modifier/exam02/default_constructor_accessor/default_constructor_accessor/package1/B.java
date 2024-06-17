package ch06_.classes.sec13.access_modifier.exam02.default_constructor_accessor.default_constructor_accessor.package1;

public class B {
	// 필드 선언
	A a1 = new A(true); 	//o
	A a2 = new A(1); 		//o
	// private 로 정의된 클래스의 생성자에는 접근불가
	//A a3 = new A("문자열");	//x
}