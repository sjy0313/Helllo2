package ch07_.inheritance.sec07_typecasting.exam02_promotion;

public class ChildExample {
	public static void main(String[] args) {
		//자식 객체 생성해서 부모 타입으로 받음
		Child child = new Child();

		//자동 타입 변환
		Parent parent = child;

		//메소드 호출
		parent.method1();
		parent.method2();
		//The method method3() is undefined for the type Parent
		//parent.method3(); //(호출 불가능)
	}
}