

package ch09_nested_class.sec02_interface_member_class.exam02;

public class AExample {
	public static void main(String[] args) {
		//A 객체 생성
		A a = new A();

		//A 인스턴스 메소드 호출
		a.useB();
		/*
		//A의 객체로 A에 속한 B의 객체를 만듬
		A.B ab = A.B(); */

	}
}