
package ch06_.classes.sec04.Student;
//s1, s2는 stack memory 에서 만들어진 로컬변수 
public class StudentExample {
	public static void main(String[] args) {
		Student s1 = new Student(); // main 함수에서 선언된 로컬변수(s1)
		System.out.println("s1 변수가 Student 객체를 참조합니다.");

		Student s2 = new Student();
		System.out.println("s2 변수가 또 다른 Student 객체를 참조합니다.");
		
		System.out.println((s1 == s2) ? "같은 객체이다" : "같은 객체가 아니다");
				
	}
}