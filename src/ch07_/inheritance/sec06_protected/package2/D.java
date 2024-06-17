package ch07_.inheritance.sec06.package2;

import ch07_.inheritance.sec06_protected.package1.A;

public class D extends A {
	//생성자 선언
	public D() {
		//A() 생성자 호출
		super();				//o
	}
	
	//메소드 선언
	public void method1() {
		//A 필드값 변경
		this.field = "value"; 	//o
		//A 메소드 호출
		this.method(); 			//o
	}
	
	//메소드 선언
	public void method2() {
		// 새로운 로컬 객체에서는 접근불가
		//A a = new A();		//x
		//a.field = "value"; 	//x
		//a.method(); 			//x
	}	
}