package ch06_.classes.sec13.access_modifier.exam02.default_constructor_accessor.default_constructor_accessor.package2;

import ch06_.classes.sec13.access_modifier.exam02.default_constructor_accessor.default_constructor_accessor.package1.*;
	
public class C {
	//필드 선언
	A a1 = new A(true); 	//o
	
	// default : 패키지가 다르므로 접근불가
	//A a2 = new A(1); 		//x
	
	// private : 같은 클래스가 아니므로 접근불가
	//A a3 = new A("문자열"); 	//x
}