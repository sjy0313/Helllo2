package ch07_.inheritance.sec07_typecasting.exam02_promotion;

public class ChildExample2 {
	public static void main(String[] args) {
		//자식 객체 생성해서 부모 타입으로 받음
		Parent parent = new Child();

		//메소드 호출
		parent.method1();
		parent.method2();
		//The method method3() is undefined for the type Parent
		//parent.method3(); //(호출 불가능)
		
		//강제 타입 변환 : 정상
		// 원래 객체는 Child 로 생성 되었기 때문에 
		Child child = (Child)parent;
		child.method3();
		
		/*
		 * Parent-method1()
Child-method2()
Child-method3()
		 */
	}
}