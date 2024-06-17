package ch09_nested_class.sec03_static_member_class.exam01;
/*
Nested class
A(outside class) $ B(member class).class 

A의 객체로 A에 속한 B의 객체를 만듬
A.B ab = A.B();*/

public class A {
	//인스턴스 멤버 클래스
	static class B {}

	//인스턴스 필드 값으로 B 객체 대입
	B field1 = new B();

	//정적 필드 값으로 B 객체 대입
	static B field2 = new B();

	//생성자
	A() {
		B b = new B();
	}

	//인스턴스 메소드
	void method1() {
		B b = new B();
	}

	//정적 메소드
	static void method2() {
		B b = new B();
	}
}